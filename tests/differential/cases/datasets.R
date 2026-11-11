# R's whole `datasets` package, compared object by object: the size of
# each dataset, the sum of every numeric column or vector (NAs removed),
# the NA count, and the level count of every factor. The values are R's
# own (written by crates/r2-base/datasets/export_datasets.R), so any
# difference is a decoding or mapping fault, not a transcription.
kv <- function(k, v) cat(k, "=", sprintf("%.17g", v), "\n", sep = "")
num <- function(k, v) { kv(paste0(k, ".sum"), sum(v, na.rm = TRUE)); kv(paste0(k, ".na"), sum(is.na(v))) }
nms <- c("ability.cov", "airmiles", "AirPassengers", "airquality", 
"anscombe", "attenu", "attitude", "austres", "beaver1", "beaver2", 
"BJsales", "BJsales.lead", "BOD", "cars", "ChickWeight", "chickwts", "co2", 
"CO2", "crimtab", "discoveries", "DNase", "esoph", "euro", "euro.cross", 
"eurodist", "EuStockMarkets", "faithful", "fdeaths", "Formaldehyde", 
"freeny", "freeny.x", "freeny.y", "gait", "HairEyeColor", "Harman23.cor", 
"Harman74.cor", "Indometh", "infert", "InsectSprays", "iris", "iris3", 
"islands", "JohnsonJohnson", "LakeHuron", "ldeaths", "lh", 
"LifeCycleSavings", "Loblolly", "longley", "lynx", "mdeaths", "morley", 
"mtcars", "nhtemp", "Nile", "nottem", "npk", "occupationalStatus", 
"Orange", "OrchardSprays", "penguins", "penguins_raw", "PlantGrowth", 
"precip", "presidents", "pressure", "Puromycin", "quakes", "randu", 
"rivers", "rock", "Seatbelts", "sleep", "stack.loss", "stack.x", 
"stackloss", "state.abb", "state.area", "state.center", "state.division", 
"state.name", "state.region", "state.x77", "sunspot.m2014", 
"sunspot.month", "sunspot.year", "sunspots", "swiss", "Theoph", "Titanic", 
"ToothGrowth", "treering", "trees", "UCBAdmissions", "UKDriverDeaths", 
"UKgas", "USAccDeaths", "USArrests", "UScitiesD", "USJudgeRatings", 
"USPersonalExpenditure", "uspop", "VADeaths", "volcano", "warpbreaks", 
"women", "WorldPhones", "WWWusage")
for (n in nms) {
  x <- get(n)
  if (is.data.frame(x)) {
    kv(paste0(n, ".nrow"), nrow(x))
    kv(paste0(n, ".ncol"), ncol(x))
    for (j in seq_len(ncol(x))) {
      col <- x[[j]]
      if (is.factor(col)) kv(paste0(n, ".", j, ".nlevels"), nlevels(col))
      else if (is.numeric(col)) num(paste0(n, ".", j), col)
    }
  } else if (is.factor(x)) {
    kv(paste0(n, ".nlevels"), nlevels(x))
  } else if (is.list(x)) {
    kv(paste0(n, ".length"), length(x))
  } else {
    kv(paste0(n, ".length"), length(x))
    if (is.numeric(x)) num(n, x)
  }
}
