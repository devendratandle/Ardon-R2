use r2_tensor::model::{Config, Model};
use r2_tensor::infer::{Sampler, SamplerConfig};
fn main() {
    // Report what the architecture scales to — same formula, real numbers.
    for (name, c) in [
        ("demo",  Config{dim:16,n_heads:4,n_kv_heads:4,n_layers:3,vocab:11,ffn_hidden:24,max_seq:32,rope_base:10000.0,eps:1e-5}),
        ("~7B",   Config{dim:4096,n_heads:32,n_kv_heads:32,n_layers:32,vocab:32000,ffn_hidden:11008,max_seq:4096,rope_base:10000.0,eps:1e-5}),
        ("~32B",  Config{dim:6656,n_heads:52,n_kv_heads:52,n_layers:60,vocab:32000,ffn_hidden:17920,max_seq:4096,rope_base:10000.0,eps:1e-5}),
    ] {
        let p = c.n_params();
        println!("{:>6}: {:>15} params  ({:.1}B)  fp16 {:.1} GB  Q4 {:.1} GB",
                 name, p, p as f64/1e9, p as f64*2.0/1e9, p as f64*0.5/1e9);
    }
    // Actually generate.
    let c = Config{dim:32,n_heads:4,n_kv_heads:2,n_layers:4,vocab:50,ffn_hidden:64,max_seq:64,rope_base:10000.0,eps:1e-5};
    let mut m = Model::zeros(c);
    let f = |n:usize,s:f32| (0..n).map(|i| ((i as f32*0.113+s).sin())*0.3).collect::<Vec<f32>>();
    m.tok_embed=f(c.vocab*c.dim,0.7); m.final_norm=vec![1.0;c.dim]; m.output=f(c.dim*c.vocab,1.9);
    for (i,l) in m.layers.iter_mut().enumerate(){ let s=i as f32*3.1;
        l.attn_norm=vec![1.0;c.dim]; l.ffn_norm=vec![1.0;c.dim];
        l.wq=f(c.dim*c.dim,s+0.2); l.wk=f(c.dim*c.kv_dim(),s+0.5); l.wv=f(c.dim*c.kv_dim(),s+0.9); l.wo=f(c.dim*c.dim,s+1.3);
        l.w1=f(c.dim*c.ffn_hidden,s+1.7); l.w2=f(c.ffn_hidden*c.dim,s+2.1); l.w3=f(c.dim*c.ffn_hidden,s+2.5); }
    let mut s = Sampler::new(42, SamplerConfig{temperature:0.9, top_k:5, ..Default::default()});
    let t0 = std::time::Instant::now();
    let out = m.generate(&[1,2,3], 40, &mut s, None).unwrap();
    let dt = t0.elapsed();
    println!("\ngenerated {} tokens in {:?} ({:.0} tok/s) -> {:?}", out.len(), dt,
             out.len() as f64/dt.as_secs_f64(), &out[..8.min(out.len())]);
}
