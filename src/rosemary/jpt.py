import os, sys, subprocess, warnings, argparse
from rosemary.tree import tree_flatten, tree_unflatten


__all__ = [
    'jpt_in_notebook',
    'jpt_autoreload',
    'jpt_full_width',
    'jpt_suppress_warnings',
    'jpt_convert_to_py',
    'jpt_tsb_stop',
    'jpt_tsb_start',
    'jpt_tsb_restart',
    'jpt_check_memusage',
    'jpt_parse_args',
    'jpt_argparse_from_config',
    'jpt_setup'
]


def jpt_setup():
    jpt_autoreload()
    jpt_full_width()


def jpt_in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            raise ImportError("vscode")
    except:
        return False
    else:  # pragma: no cover
        return True


def jpt_autoreload():
    if jpt_in_notebook():
        from IPython.core.getipython import get_ipython
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')


def jpt_full_width():
    if jpt_in_notebook():
        from IPython.core.display import display, HTML
        display(HTML("<style>.container { width:100% !important; }</style>"))


def jpt_suppress_warnings():
    from pandas.core.common import SettingWithCopyWarning
    if jpt_in_notebook():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", SettingWithCopyWarning)
    else:
        warnings.filterwarnings("ignore")


def jpt_convert_to_py(ipynb_path):
    if jpt_in_notebook():
        import shlex
        cmd = f"jupyter nbconvert --to script {ipynb_path}"
        try:
            retcode = subprocess.call(shlex.split(cmd))
        except OSError as err:
            print(f'Execution failed: [{cmd}]')


def jpt_tsb_stop(port):
    # Use SIGTERM/15 instead of SIGKILL/9 so `tensorboard` does proper clean-ups
    os.system(f'kill -15 $(lsof -t -i:{port})')


def jpt_tsb_start(logdir, port):
    """Does same thing as `tensorboard` extension as Jupyter magics
            %tensorboard --logdir=$logdir --port=$port
    """
    from tensorboard import notebook
    notebook.start(f'--logdir="{logdir}" --port={port}')


def jpt_tsb_restart(logdir, port):
    if jpt_in_notebook():
        jpt_tsb_stop(port)
        jpt_tsb_start(logdir, port)


def jpt_check_memusage(globals_, dir_, getsize_opt='pympler', in_gb=True):
    """ Check memory usage of global variables 
        utils.jpt_check_memusage(globals(), dir())
    """
    from pympler import asizeof

    if getsize_opt == 'pympler':
        getsize = asizeof.asizeof
    else:
        getsize = sys.getsizeof

    if jpt_in_notebook():
        # These are the usual ipython objects
        ipython_vars = ['In', 'Out', 'exit',
                        'quit', 'get_ipython', 'ipython_vars']
    else:
        ipython_vars = []

    l = []
    b = []
    for x in dir_:
        if x.startswith('_') or x in sys.modules or x in ipython_vars:
            continue
        try:
            size = getsize(globals_.get(x))
            if in_gb:
                size = size/1024**3
            l.append((x, size))
        except:
            b.append(x)
    l = sorted(l, key=lambda x: x[1], reverse=True)
    return l, b


def jpt_split_cmd(cmd=None):
    import shlex
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    return cmd


def jpt_parse_args(parser, cmd=None):
    """"
    args = jpt_parse_args(parser, args='--argument=value')
    globals().update(args.__dict__) """""
    if jpt_in_notebook() and cmd is not None:
        return parser.parse_args(jpt_split_cmd(cmd))
    else:
        return parser.parse_args()


def jpt_parse_known_args(parser, cmd=None):
    if jpt_in_notebook() and cmd is not None:
        return parser.parse_known_args(jpt_split_cmd(cmd))
    else:
        return parser.parse_known_args()


def jpt_argparse_from_config(cmd=None,
                             parser_addargs_callback=lambda parser: None):
    """Construct argparse parser from config files so that
            - read in config file to populate argparse arguments
            - modify fields in config files from command line arguments  
            - specify new arguments with `add_argument_callback`

        cfg = jpt_argparse_from_config(cmd="--config=config.yaml",
                                       parser_addargs_callback)
    """
    from omegaconf import OmegaConf
    
    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False)

    conf_parser.add_argument("-c", "--config", 
                             metavar="FILE")
    conf_args, remaining_argv = jpt_parse_known_args(
        conf_parser, cmd)
    
    cfg = OmegaConf.load(conf_args.config) if conf_args.config \
        else OmegaConf.create()

    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        parents=[conf_parser])

    for k, v in tree_flatten(cfg).items():
        if type(v) == bool:
            parser.add_argument(f'--{k}', default=v, action='store_true')
            parser.add_argument(f'--no-{k}', dest=k, action='store_false')
        else:
            parser.add_argument(f'--{k}', type=type(v), default=v)

    parser_addargs_callback(parser)
    
    args = parser.parse_args(remaining_argv)
    args.config = conf_args.config

    args_nested = tree_unflatten(vars(args))
    cfg = OmegaConf.create(args_nested)
            
    return cfg