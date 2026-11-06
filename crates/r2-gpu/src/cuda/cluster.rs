//! Synchronous training over several processes, one GPU each: the mean of every
//! rank's gradient, the same bits on every rank.
//!
//! Rank 0 takes the others' gradients in rank order, sums them in that order and
//! sends the one result back, so the arithmetic depends on neither timing nor
//! topology and the ranks' weights cannot drift. Plain TCP from std, so the same
//! code runs between two cards in a machine and between machines. It is a star,
//! which is what two ranks need; rank 0's links are the limit at many ranks.

use super::ffi::{HostBuffer, SharedMem};
use super::{optim, Tensor};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// How long a silent peer is waited for before the step fails.
const SILENCE: Duration = Duration::from_secs(600);

/// Seconds an exchange spent, summed since the start.
#[derive(Clone, Copy, Debug, Default)]
pub struct Times { pub device_copies: f64, pub network: f64, pub averaging: f64, pub exchanges: u64 }

pub struct Cluster {
    pub rank: usize,
    pub world: usize,
    /// Rank 0: ranks 1, 2, ... in order. Others: rank 0.
    peers: Vec<TcpStream>,
    addr: String,
    mine: Option<HostBuffer>,
    theirs: Option<HostBuffer>,
    decided: bool,
    shared: Option<Shared>,
    why_tcp: Option<String>,
    pub times: Times,
}

/// The fast road between two ranks on one machine.
struct Shared { mine: SharedMem, theirs: SharedMem, scratch: Tensor }

fn io(what: &str) -> impl Fn(std::io::Error) -> String + '_ { move |e| format!("cluster {what}: {e}") }

fn configured(s: TcpStream) -> Result<TcpStream, String> {
    s.set_nodelay(true).map_err(io("nodelay"))?;
    s.set_read_timeout(Some(SILENCE)).map_err(io("timeout"))?;
    s.set_write_timeout(Some(SILENCE)).map_err(io("timeout"))?;
    Ok(s)
}

impl Cluster {
    /// `R2_WORLD`, `R2_RANK` and `R2_CLUSTER_ADDR` (default 127.0.0.1:29500).
    /// `None` when there is one rank.
    pub fn from_env() -> Result<Option<Cluster>, String> {
        let num = |k: &str, d: usize| std::env::var(k).ok().map_or(Ok(d), |v| v.parse().map_err(|_| format!("{k}: {v:?}")));
        let (world, rank) = (num("R2_WORLD", 1)?, num("R2_RANK", 0)?);
        if world <= 1 { return Ok(None); }
        let addr = std::env::var("R2_CLUSTER_ADDR").unwrap_or_else(|_| "127.0.0.1:29500".into());
        Cluster::connect(rank, world, &addr, Duration::from_secs(300)).map(Some)
    }

    /// Rank 0 listens at `addr`; each other rank connects and states its rank.
    pub fn connect(rank: usize, world: usize, addr: &str, wait: Duration) -> Result<Cluster, String> {
        if rank >= world { return Err(format!("rank {rank} in a world of {world}")); }
        let start = Instant::now();
        let peers = if rank == 0 {
            let l = TcpListener::bind(addr).map_err(io("bind"))?;
            l.set_nonblocking(true).map_err(io("listen"))?;
            let mut slots: Vec<Option<TcpStream>> = (1..world).map(|_| None).collect();
            while slots.iter().any(Option::is_none) {
                match l.accept() {
                    Ok((s, _)) => {
                        s.set_nonblocking(false).map_err(io("accept"))?;
                        let mut s = configured(s)?;
                        let mut b = [0u8; 8];
                        s.read_exact(&mut b).map_err(io("hello"))?;
                        let r = u64::from_le_bytes(b) as usize;
                        let slot = r.checked_sub(1).and_then(|i| slots.get_mut(i)).ok_or(format!("hello from rank {r}"))?;
                        *slot = Some(s);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if start.elapsed() > wait { return Err(format!("cluster: not all of {world} ranks came within {wait:?}")); }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(e) => return Err(format!("cluster accept: {e}")),
                }
            }
            slots.into_iter().flatten().collect()
        } else {
            let mut s = loop {
                match TcpStream::connect(addr) {
                    Ok(s) => break configured(s)?,
                    Err(e) if start.elapsed() > wait => return Err(format!("cluster connect to {addr}: {e}")),
                    Err(_) => std::thread::sleep(Duration::from_millis(50)),
                }
            };
            s.write_all(&(rank as u64).to_le_bytes()).map_err(io("hello"))?;
            vec![s]
        };
        Ok(Cluster { rank, world, addr: addr.to_string(), peers, mine: None, theirs: None, decided: false, shared: None, why_tcp: None, times: Times::default() })
    }

    /// Replaces each `grads[i]`'s first `sizes[i]` elements with the mean over the
    /// ranks. Every rank must call this with the same sizes; waits for kernels
    /// launched before it.
    pub fn average(&mut self, grads: &[&Tensor], sizes: &[usize]) -> Result<(), String> {
        let total: usize = sizes.iter().sum();
        if !self.decided {
            self.decided = true;
            self.shared = self.try_shared(total)?;
            println!("  cluster    gradients through {}", match &self.shared {
                Some(_) => "shared memory, averaged on each GPU".to_string(),
                None => format!("TCP{}", self.why_tcp.as_deref().map_or(String::new(), |w| format!(" ({w})"))),
            });
        }
        if self.shared.is_some() { return self.average_shared(grads, sizes); }
        if self.mine.as_ref().is_none_or(|b| b.as_slice().len() != total) {
            self.mine = Some(HostBuffer::new(total)?);
            if self.rank == 0 { self.theirs = Some(HostBuffer::new(total)?); }
        }
        let mine = self.mine.as_mut().ok_or("cluster buffer")?;
        let t = Instant::now();
        let mut at = 0;
        for (g, &n) in grads.iter().zip(sizes) {
            g.copy_to(&mut mine.as_mut_slice()[at..at + n])?;
            at += n;
        }
        self.times.device_copies += t.elapsed().as_secs_f64();

        if self.rank == 0 {
            let theirs = self.theirs.as_mut().ok_or("cluster buffer")?;
            for p in &mut self.peers {
                let t = Instant::now();
                p.read_exact(theirs.as_mut_bytes()).map_err(io("receive"))?;
                self.times.network += t.elapsed().as_secs_f64();
                let t = Instant::now();
                for (a, b) in mine.as_mut_slice().iter_mut().zip(theirs.as_slice()) { *a += *b; }
                self.times.averaging += t.elapsed().as_secs_f64();
            }
            let t = Instant::now();
            let inv = 1.0 / self.world as f32;
            for a in mine.as_mut_slice() { *a *= inv; }
            self.times.averaging += t.elapsed().as_secs_f64();
            let t = Instant::now();
            for p in &mut self.peers { p.write_all(mine.as_bytes()).map_err(io("send"))?; }
            self.times.network += t.elapsed().as_secs_f64();
        } else {
            let t = Instant::now();
            self.peers[0].write_all(mine.as_bytes()).map_err(io("send"))?;
            self.peers[0].read_exact(mine.as_mut_bytes()).map_err(io("receive"))?;
            self.times.network += t.elapsed().as_secs_f64();
        }

        let t = Instant::now();
        let mut at = 0;
        for (g, &n) in grads.iter().zip(sizes) {
            g.copy_from(0, &mine.as_slice()[at..at + n])?;
            at += n;
        }
        self.times.device_copies += t.elapsed().as_secs_f64();
        self.times.exchanges += 1;
        Ok(())
    }

    /// Two ranks on one machine meet in `/dev/shm`. Both must succeed, or both use
    /// TCP: `R2_CLUSTER_SHM=0` asks for that, as does an address that is not local.
    fn try_shared(&mut self, total: usize) -> Result<Option<Shared>, String> {
        let local = self.addr.starts_with("127.") || self.addr.starts_with("localhost");
        let wanted = self.world == 2 && local && std::env::var("R2_CLUSTER_SHM").as_deref() != Ok("0");
        if !self.agree(wanted as u64)? || !wanted {
            self.why_tcp = Some("not two ranks on one machine, or asked for".into());
            return Ok(None);
        }
        let port: String = self.addr.chars().filter(char::is_ascii_alphanumeric).collect();
        let path = |r: usize| format!("/dev/shm/r2-cluster-{port}-{r}");
        let mine = SharedMem::map(&path(self.rank), total, true);
        self.agree(1)?; // both files exist, or their failure shows below
        let theirs = SharedMem::map(&path(1 - self.rank), total, false);
        let scratch = Tensor::zeros(total);
        let ok = mine.is_ok() && theirs.is_ok() && scratch.is_some();
        let all = self.agree(ok as u64)?;
        let _ = std::fs::remove_file(path(self.rank));
        match (all && ok, mine, theirs, scratch) {
            (true, Ok(mine), Ok(theirs), Some(scratch)) => Ok(Some(Shared { mine, theirs, scratch })),
            (_, m, t, _) => {
                self.why_tcp = Some(m.err().or(t.err()).unwrap_or_else(|| "the other rank could not".into()));
                Ok(None)
            }
        }
    }

    /// Each rank copies its gradient out, then the other's in, and averages on its
    /// own GPU: no host arithmetic, and the same bits on both, as addition commutes.
    fn average_shared(&mut self, grads: &[&Tensor], sizes: &[usize]) -> Result<(), String> {
        let sh = self.shared.as_mut().ok_or("cluster buffer")?;
        super::sync();
        let t = Instant::now();
        let mut at = 0;
        for (g, &n) in grads.iter().zip(sizes) {
            g.copy_to(&mut sh.mine.as_mut_slice()[at..at + n])?;
            at += n;
        }
        self.times.device_copies += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let peer = &mut self.peers[0];
        peer.write_all(&[1]).map_err(io("ready"))?;
        peer.read_exact(&mut [0u8; 1]).map_err(io("ready"))?;
        self.times.network += t.elapsed().as_secs_f64();
        let t = Instant::now();
        sh.scratch.copy_from(0, sh.theirs.as_slice())?;
        self.times.device_copies += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut at = 0;
        for (g, &n) in grads.iter().zip(sizes) {
            if !optim::grad_mean(g, &sh.scratch, at, n) { return Err("GPU kernel failed: gradient mean".into()); }
            at += n;
        }
        // before the other rank may overwrite what it handed over
        super::sync();
        self.times.averaging += t.elapsed().as_secs_f64();
        let t = Instant::now();
        peer.write_all(&[2]).map_err(io("done"))?;
        peer.read_exact(&mut [0u8; 1]).map_err(io("done"))?;
        self.times.network += t.elapsed().as_secs_f64();
        self.times.exchanges += 1;
        Ok(())
    }

    /// Whether every rank holds the same `value`, and a barrier: no rank returns
    /// before all have called this.
    pub fn agree(&mut self, value: u64) -> Result<bool, String> {
        if self.rank == 0 {
            let mut same = true;
            for p in &mut self.peers {
                let mut b = [0u8; 8];
                p.read_exact(&mut b).map_err(io("agree"))?;
                same &= u64::from_le_bytes(b) == value;
            }
            for p in &mut self.peers { p.write_all(&[same as u8]).map_err(io("agree"))?; }
            Ok(same)
        } else {
            self.peers[0].write_all(&value.to_le_bytes()).map_err(io("agree"))?;
            let mut b = [0u8; 1];
            self.peers[0].read_exact(&mut b).map_err(io("agree"))?;
            Ok(b[0] == 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::have_device;

    /// Three ranks in threads on one device: the mean, the same bits on all.
    #[test]
    fn every_rank_gets_the_same_mean() {
        if !have_device() { return; }
        let addr = "127.0.0.1:29517";
        let wait = Duration::from_secs(30);
        let run = |rank: usize| std::thread::spawn(move || {
            let mut c = Cluster::connect(rank, 3, addr, wait).unwrap();
            let r = rank as f32;
            let (a, b) = (Tensor::upload(&[r, 0.1 * r, 1.0, 7.0]).unwrap(), Tensor::upload(&[r * r, 9.0]).unwrap());
            c.average(&[&a, &b], &[3, 1]).unwrap();
            assert!(c.agree(11).unwrap());
            (a.to_vec().unwrap(), b.to_vec().unwrap())
        });
        let hs: Vec<_> = (0..3).map(run).collect();
        let out: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(out[0], out[1]);
        assert_eq!(out[0], out[2]);
        assert_eq!(out[0].0[0], 1.0);
        assert_eq!(out[0].0[3], 7.0, "past sizes[i] is untouched");
        assert_eq!(out[0].1, vec![(0.0 + 1.0 + 4.0) * (1.0f32 / 3.0), 9.0]);
    }

    /// Two ranks through shared memory: the same mean, the same bits, on one device.
    #[test]
    fn two_ranks_share_memory_and_agree() {
        if !have_device() { return; }
        let run = |rank: usize| std::thread::spawn(move || {
            let mut c = Cluster::connect(rank, 2, "127.0.0.1:29520", Duration::from_secs(30)).unwrap();
            let r = rank as f32;
            let (a, b) = (Tensor::upload(&[r, 0.1 + r, 3.0, 7.0]).unwrap(), Tensor::upload(&[1e-3 * r, 9.0]).unwrap());
            for _ in 0..2 { c.average(&[&a, &b], &[3, 1]).unwrap(); }
            assert!(c.shared.is_some(), "shared memory is available here: {:?}", c.why_tcp);
            (a.to_vec().unwrap(), b.to_vec().unwrap())
        });
        let (x, y) = (run(0), run(1));
        let (x, y) = (x.join().unwrap(), y.join().unwrap());
        assert_eq!(x, y);
        assert_eq!(x.0[0], 0.5);
        assert_eq!(x.0[3], 7.0, "past sizes[i] is untouched");
        assert_eq!(x.1[1], 9.0);
    }

    #[test]
    fn disagreement_is_reported_to_every_rank() {
        if !have_device() { return; }
        let addr = "127.0.0.1:29518";
        let run = |rank: usize| std::thread::spawn(move || {
            Cluster::connect(rank, 2, addr, Duration::from_secs(30)).unwrap().agree(rank as u64).unwrap()
        });
        let (a, b) = (run(0), run(1));
        assert!(!a.join().unwrap() && !b.join().unwrap());
    }

    #[test]
    fn missing_ranks_fail_and_do_not_hang() {
        let e = Cluster::connect(0, 2, "127.0.0.1:29519", Duration::from_millis(100)).err().unwrap();
        assert!(e.contains("came within"), "{e}");
    }
}
