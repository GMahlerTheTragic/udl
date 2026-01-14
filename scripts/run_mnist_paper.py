from udl.al_mnist.runner import run
from udl.utils.utils import (
    dir_atifacts,
    make_run_paths_config,
    setup_huggingface_stuff,
    initialize_logging,
    step_rows_to_csv,
)


def main() -> int:
    initialize_logging()
    rp = make_run_paths_config()
    setup_huggingface_stuff(artifacts=dir_atifacts())
    results = run()
    step_rows_to_csv(rp.csv_output_path, results)
    return 0


if __name__ == "__main__":
    raise main()
