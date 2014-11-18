library('ForeCA')

args <- commandArgs(trailingOnly = TRUE)
X <- read.csv(sprintf('example_2014_11_foreca_train_%s.csv', args), header=FALSE)
ff <- foreca(X, n.comp=2)
W <- matrix(ff$loadings, ncol=2)
write.matrix(W, sprintf('example_2014_11_foreca_result_%s.csv', args))
