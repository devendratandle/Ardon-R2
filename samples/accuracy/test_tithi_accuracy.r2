# Tithi accuracy check against well-known 2024 Hindu festival dates.

check <- function(date_str, label) {
  d <- as.Date(date_str)
  h <- hnc.date(d)
  cat(label, " | ", date_str, " -> ", h$formatted, "\n", sep="")
}

cat("\n=== 2024 Festival Tithi Verification ===\n")
cat("Format: SSSS-MM-P-TT  (Saka year - month - paksha[1=Shukla,2=Krishna] - tithi)\n")
cat("Expected lunar dates listed in comments next to each call.\n\n")

# Holika Dahan = Phalguna Shukla Purnima (full moon)
check("2024-03-24", "Holika Dahan (expect Phalguna Pur):    ")
# Dhulendi/Holi = Chaitra Krishna Pratipada (in amanta reckoning)
check("2024-03-25", "Dhulendi / Holi (expect 01-2-01):      ")
# Gudi Padwa = Chaitra Shukla Pratipada  (HNC New Year)
check("2024-04-09", "Gudi Padwa (expect 01-1-01):           ")
# Ram Navami = Chaitra Shukla Navami (9)
check("2024-04-17", "Ram Navami (expect 01-1-09):           ")
# Akshaya Tritiya = Vaisakha Shukla Tritiya (3)
check("2024-05-10", "Akshaya Tritiya (expect 02-1-03):      ")
# Guru Purnima = Ashadha Shukla Purnima
check("2024-07-21", "Guru Purnima (expect 04-1-15):         ")
# Raksha Bandhan = Shravana Shukla Purnima
check("2024-08-19", "Raksha Bandhan (expect 05-1-15):       ")
# Krishna Janmashtami = Bhadrapada Krishna Ashtami (amanta = Shravana K8 purnimanta)
check("2024-08-26", "Janmashtami (expect 06-2-08):          ")
# Ganesh Chaturthi = Bhadrapada Shukla Chaturthi
check("2024-09-07", "Ganesh Chaturthi (expect 06-1-04):     ")
# Sharad Purnima = Ashwin Shukla Purnima
check("2024-10-16", "Sharad Purnima (expect 07-1-15):       ")
# Diwali = Kartika (amanta: Ashwin K30) Krishna Amavasya
check("2024-11-01", "Diwali (expect 07-2-15):               ")
