library('ForeCA')

args <- commandArgs(trailingOnly = TRUE)
arg_id <- args[1]
arg_dim <- as.numeric(args[2])

X <- read.csv(sprintf('foreca_node_train_%s.csv', arg_id), header=FALSE)
ff <- foreca(X, n.comp=arg_dim)
summary(ff)
print(ff$loadings)
W <- matrix(ff$loadings, ncol=arg_dim)

write.matrix(W, sprintf('foreca_node_result_%s.csv', arg_id))
