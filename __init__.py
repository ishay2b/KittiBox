
def git_root():
    ''' return the root location of git rep
    '''
    import subprocess
    gitroot = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    return gitroot

