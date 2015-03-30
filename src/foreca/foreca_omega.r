library('ForeCA')

args <- commandArgs(trailingOnly = TRUE)
arg_id <- args[1]
arg_dim <- as.numeric(args[2])

X <- ts(read.csv(sprintf('foreca_omega_input_%s.csv', arg_id), header=FALSE))
O <- Omega(X, spectrum.control = list(method = "wosa"))
print(O)
M <- matrix(O)

write.matrix(M, sprintf('foreca_omega_result_%s.csv', arg_id))
