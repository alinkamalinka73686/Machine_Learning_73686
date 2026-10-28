import logging

from sklearn.datasets import fetch_openml

log = logging.getLogger("ingest")

EXPECTED_ROWS = 48842
EXPECTED_COLS = 15


def load_raw():
    bunch = fetch_openml(name="adult", version=2, as_frame=True)
    df = bunch.frame.copy()

    target = bunch.target_names[0]
    if target != "income":
        df = df.rename(columns={target: "income"})

    rows, cols = df.shape
    log.info("Raw data loaded: %d rows, %d columns", rows, cols)

    if rows != EXPECTED_ROWS or cols != EXPECTED_COLS:
        log.warning(
            "Unexpected shape: got (%d, %d), expected (%d, %d)",
            rows, cols, EXPECTED_ROWS, EXPECTED_COLS,
        )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_raw()
