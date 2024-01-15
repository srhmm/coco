from enum import Enum


class CoCoTestType(Enum):
    ''' Test for confounding between pairs of variables
    '''
    SKIP = -1
    MI_ZTEST = 0
    AMI_ZTEST = 1
    VI_ZTEST = 2
    MI = 3
    AMI = 4
    VI = 5


class CoDAGType(Enum):
    ''' Method for discovering a causal DAG
    '''
    SKIP = 0 # oracle
    MSS = 1

class CoShiftTestType(Enum):
    ''' Test for mechanism shifts between pairs of contexts
    '''
    #skip (e.g. if we only need oracle results)
    SKIP = 0
    # clustering
    VARIO = 1
    VARIO_GREEDY = 2
    LINC = 3
    # KCI pairwise- main used
    PI_KCI = 4
    #PI_GP = 5
    #PI_LINEAR = 6
    #SOFT_KCI = 7
    #SOFT_GP = 8
    #SOFT_OT = 9
    #SOFT_CMMD = 10
