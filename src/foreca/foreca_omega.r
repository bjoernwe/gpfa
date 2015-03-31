library('ForeCA')

print(getwd())

args <- commandArgs(trailingOnly = TRUE)
arg_id <- args[1]
arg_cwd <- args[2]

X <- ts(read.csv(sprintf('%s/foreca_omega_input_%s.csv', arg_cwd, arg_id), header=FALSE))
O <- Omega(X, spectrum.control = list(method = "wosa"))
print(O)
M <- matrix(O)

write.matrix(M, sprintf('%s/foreca_omega_result_%s.csv', arg_cwd, arg_id))
