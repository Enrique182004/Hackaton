from biobench_sar.envfile import load_env_file

load_env_file()  # before app import: Settings reads the environment when it is built

from biobench_sar.app import main  # noqa: E402

main()
