import numpy as np
import os

from subprocess import call


def omega(x):
    
    # save training data in CSV
    cwd = os.getcwd()
    dir = os.path.dirname(os.path.abspath(__file__))
    run_id = str(np.random.randint(100000, 1000000))
    np.savetxt("%s/foreca_omega_input_%s.csv" % (cwd, run_id), x, delimiter=",")
    
    # run R script and load result
    call(['Rscript', '%s/foreca_omega.r' % dir, run_id, cwd])
    O = np.loadtxt('%s/foreca_omega_result_%s.csv' % (cwd, run_id))

    # clean files
    call(['rm', '%s/foreca_omega_input_%s.csv' % (cwd, run_id)])
    call(['rm', '%s/foreca_omega_result_%s.csv' % (cwd, run_id)])

    return O


if __name__ == '__main__':
    
    print omega(np.random.randn(1000,3))