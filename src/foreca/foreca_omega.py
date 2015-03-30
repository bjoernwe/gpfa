import numpy as np

from subprocess import call


def omega(x):
    
    # save training data in CSV
    run_id = str(np.random.randint(100000, 1000000))
    np.savetxt("foreca_omega_input_%s.csv" % run_id, x, delimiter=",")
    
    # run R script and load result
    call(['Rscript', 'foreca_omega.r', run_id])#, str(output_dim)])
    O = np.loadtxt('foreca_omega_result_%s.csv' % run_id)

    # clean files
    call(['rm', 'foreca_omega_input_%s.csv' % run_id])
    call(['rm', 'foreca_omega_result_%s.csv' % run_id])

    return O


if __name__ == '__main__':
    
    print omega(np.random.randn(1000,3))