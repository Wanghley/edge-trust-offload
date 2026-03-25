import argparse
import os
import sys

import h5py
import pandas as pd


def extract_participant_slice(hdf5_path, participant_id, task_name, output_dir, overwrite=False):
    """
    Extract one participant/task dataset from an HDF5 file and save it as CSV.
    """
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5_file:
        if participant_id not in h5_file:
            available = sorted(h5_file.keys())
            sample = ", ".join(available[:10])
            raise KeyError(
                f"Participant '{participant_id}' not found. "
                f"Available participants (first 10): {sample}"
            )

        participant_group = h5_file[participant_id]
        if task_name not in participant_group:
            available_tasks = sorted(participant_group.keys())
            sample_tasks = ", ".join(available_tasks[:10])
            raise KeyError(
                f"Task '{task_name}' not found for participant '{participant_id}'. "
                f"Available tasks (first 10): {sample_tasks}"
            )

        dataset = participant_group[task_name]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(
                f"Path '{participant_id}/{task_name}' is not a dataset and cannot be exported to CSV."
            )

        data = dataset[:]
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        columns = None
        if dataset.dtype.names:
            columns = list(dataset.dtype.names)

        dataframe = pd.DataFrame(data, columns=columns)

        output_path = os.path.join(output_dir, f"{participant_id}_{task_name}.csv")
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. Use --overwrite to replace it."
            )

        dataframe.to_csv(output_path, index=False)
        return output_path, dataframe.shape


def list_participants(hdf5_path):
    """Return participant IDs from the root level of the HDF5 file."""
    with h5py.File(hdf5_path, "r") as h5_file:
        return sorted(h5_file.keys())


def list_tasks(hdf5_path, participant_id):
    """Return task names for the provided participant."""
    with h5py.File(hdf5_path, "r") as h5_file:
        if participant_id not in h5_file:
            raise KeyError(f"Participant '{participant_id}' not found.")
        return sorted(h5_file[participant_id].keys())


def download_dataset(dataset_ref):
    """Download a dataset from KaggleHub and return the local path."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        ) from exc

    return kagglehub.dataset_download(dataset_ref)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract participant/task slices from a biomechanics HDF5 file to CSV."
    )
    parser.add_argument(
        "--hdf5-path",
        default="data/raw/BiomechanicsDataset.hdf5",
        help="Path to input HDF5 file (default: data/raw/BiomechanicsDataset.hdf5)",
    )
    parser.add_argument(
        "--participant-id",
        help="Participant ID to export, e.g., B026",
    )
    parser.add_argument(
        "--task-name",
        help="Task name to export, e.g., IndexFlexion_MVC",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where CSV output will be written (default: data/processed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSV if present.",
    )
    parser.add_argument(
        "--list-participants",
        action="store_true",
        help="List participant IDs in the HDF5 file and exit.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List tasks for --participant-id and exit.",
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download the dataset from KaggleHub and exit.",
    )
    parser.add_argument(
        "--dataset-ref",
        default="maximilliantdiaz/bham-biomechanics-hand-modeling-dataset",
        help=(
            "Kaggle dataset reference in owner/dataset format "
            "(default: maximilliantdiaz/bham-biomechanics-hand-modeling-dataset)"
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.download_dataset:
            download_path = download_dataset(args.dataset_ref)
            print(f"Path to dataset files: {download_path}")
            return 0

        if args.list_participants:
            participants = list_participants(args.hdf5_path)
            print("Participants:")
            for participant in participants:
                print(participant)
            return 0

        if args.list_tasks:
            if not args.participant_id:
                parser.error("--list-tasks requires --participant-id")
            tasks = list_tasks(args.hdf5_path, args.participant_id)
            print(f"Tasks for {args.participant_id}:")
            for task in tasks:
                print(task)
            return 0

        if not args.participant_id or not args.task_name:
            parser.error(
                "Extraction requires --participant-id and --task-name, or use a --list-* option."
            )

        output_path, shape = extract_participant_slice(
            hdf5_path=args.hdf5_path,
            participant_id=args.participant_id,
            task_name=args.task_name,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(f"Saved CSV to: {output_path}")
        print(f"Rows x Cols: {shape[0]} x {shape[1]}")
        return 0

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        FileExistsError,
        OSError,
        ValueError,
        ImportError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())