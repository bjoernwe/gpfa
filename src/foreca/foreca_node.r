library('ForeCA')

args <- commandArgs(trailingOnly = TRUE)
arg_id <- args[1]
arg_dim <- as.numeric(args[2])
arg_cwd <- args[3]

X <- ts(read.csv(sprintf('%s/foreca_node_train_%s.csv', arg_cwd, arg_id), header=FALSE))
ff <- foreca(X, n.comp=arg_dim)
summary(ff)
print(ff$loadings)
W <- matrix(ff$loadings, ncol=arg_dim)

write.matrix(W, sprintf('%s/foreca_node_result_%s.csv', arg_cwd, arg_id))
