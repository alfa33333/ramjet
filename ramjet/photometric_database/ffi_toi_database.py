"""
Code to represent a database to train to find exoplanet transits in FFI data based on known TOI dispositions.
"""
from functools import partial

import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Iterable
from pathlib import Path

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.data_interface.tess_ffi_data_interface import TessFfiDataInterface
from ramjet.data_interface.tess_toi_data_interface import ToiColumns, TessToiDataInterface
from ramjet.photometric_database.injected_with_additional_explicit_injected_negative_database import \
    InjectedWithAdditionalExplicitInjectedNegativeDatabase
from ramjet.photometric_database.tess_synthetic_injected_with_negative_injection_database import \
    TessSyntheticInjectedWithNegativeInjectionDatabase
from ramjet.py_mapper import map_py_function_to_dataset


class FfiToiDatabase(InjectedWithAdditionalExplicitInjectedNegativeDatabase):
    """
    Code to represent a database to train to find exoplanet transits in FFI data based on known TOI dispositions.
    """
    def __init__(self, data_directory='data/toi_ffi_anti_eclipsing_binary_database'):
        super().__init__(data_directory=data_directory)
        self.toi_dispositions_path = self.data_directory.joinpath('toi_dispositions.csv')
        self.time_steps_per_example = 1296  # 27 days / 30 minutes.
        self.batch_size = 1000
        self.allow_out_of_bounds_injection = True
        self.tess_ffi_data_interface = TessFfiDataInterface()
        self.tess_toi_data_interface = TessToiDataInterface()

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets for the database.

        :return: The training and validation dataset.
        """
        synthetic_signal_paths_dataset = self.paths_dataset_from_list_or_generator_factory(
            self.get_all_synthetic_signal_paths)
        training_lightcurve_path_generator = partial(self.tess_ffi_data_interface.paths_generator_from_sql_table,
            dataset_splits=list(range(8)), magnitudes=[9]
        )
        training_lightcurve_paths_dataset = self.paths_dataset_from_list_or_generator_factory(
            training_lightcurve_path_generator)
        validation_lightcurve_path_generator = partial(self.tess_ffi_data_interface.paths_generator_from_sql_table,
            dataset_splits=[8], magnitudes=[9]
        )
        validation_lightcurve_paths_dataset = self.paths_dataset_from_list_or_generator_factory(
            validation_lightcurve_path_generator)
        negative_synthetic_signal_paths = self.paths_dataset_from_list_or_generator_factory(
            self.get_all_negative_synthetic_signal_paths)
        explicit_negative_synthetic_signal_paths = self.paths_dataset_from_list_or_generator_factory(
            self.get_explicit_negative_synthetic_signal_paths)
        shuffled_training_lightcurve_paths_dataset = training_lightcurve_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        shuffled_synthetic_signal_paths_dataset = synthetic_signal_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        shuffled_negative_synthetic_signal_paths_dataset = negative_synthetic_signal_paths.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        shuffled_explicit_negative_synthetic_paths_dataset = explicit_negative_synthetic_signal_paths.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        zipped_training_paths_dataset = tf.data.Dataset.zip((shuffled_training_lightcurve_paths_dataset,
                                                             shuffled_synthetic_signal_paths_dataset,
                                                             shuffled_negative_synthetic_signal_paths_dataset,
                                                             shuffled_explicit_negative_synthetic_paths_dataset))
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, 1), (1,)]
        lightcurve_training_dataset = map_py_function_to_dataset(zipped_training_paths_dataset,
                                                                 self.positive_injection_negative_injection_and_explicit_negative_injection_preprocessing,
                                                                 self.number_of_parallel_processes_per_map,
                                                                 output_types=output_types,
                                                                 output_shapes=output_shapes,
                                                                 flat_map=True)
        batched_training_dataset = self.window_dataset_for_zipped_example_and_label_dataset(lightcurve_training_dataset,
                                                                                            self.batch_size,
                                                                                            self.batch_size // 10)
        prefetch_training_dataset = batched_training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        shuffled_validation_lightcurve_paths_dataset = validation_lightcurve_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        zipped_validation_paths_dataset = tf.data.Dataset.zip((shuffled_validation_lightcurve_paths_dataset,
                                                               shuffled_synthetic_signal_paths_dataset,
                                                               shuffled_negative_synthetic_signal_paths_dataset,
                                                               shuffled_explicit_negative_synthetic_paths_dataset))
        lightcurve_validation_dataset = map_py_function_to_dataset(zipped_validation_paths_dataset,
                                                                   self.positive_injection_negative_injection_and_explicit_negative_injection_preprocessing,
                                                                   self.number_of_parallel_processes_per_map,
                                                                   output_types=output_types,
                                                                   output_shapes=output_shapes,
                                                                   flat_map=True)
        batched_validation_dataset = lightcurve_validation_dataset.batch(self.batch_size)
        prefetch_validation_dataset = batched_validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return prefetch_training_dataset, prefetch_validation_dataset

    def download_exofop_toi_lightcurves_to_synthetic_directory(self):
        """
        Downloads the `ExoFOP database <https://exofop.ipac.caltech.edu/tess/view_toi.php>`_ lightcurve files to the
        synthetic directory.
        """
        print("Downloading ExoFOP TOI disposition CSV...")
        self.create_data_directories()
        toi_csv_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
        response = requests.get(toi_csv_url)
        with self.toi_dispositions_path.open('wb') as csv_file:
            csv_file.write(response.content)
        toi_dispositions = self.tess_toi_data_interface.toi_dispositions
        tic_ids = toi_dispositions[ToiColumns.tic_id.value].unique()
        print('Downloading TESS obdservation list...')
        tess_data_interface = TessDataInterface()
        tess_observations = tess_data_interface.get_all_tess_time_series_observations(tic_id=tic_ids)
        single_sector_observations = tess_data_interface.filter_for_single_sector_observations(tess_observations)
        single_sector_observations = tess_data_interface.add_tic_id_column_to_single_sector_observations(
            single_sector_observations)
        single_sector_observations = tess_data_interface.add_sector_column_to_single_sector_observations(
            single_sector_observations)
        print("Downloading lightcurves which are confirmed or suspected planets in TOI dispositions...")
        suspected_planet_dispositions = toi_dispositions[toi_dispositions[ToiColumns.disposition.value] != 'FP']
        suspected_planet_observations = pd.merge(single_sector_observations, suspected_planet_dispositions, how='inner',
                                                 on=[ToiColumns.tic_id.value, ToiColumns.sector.value])
        observations_not_found = suspected_planet_dispositions.shape[0] - suspected_planet_observations.shape[0]
        print(f"{suspected_planet_observations.shape[0]} observations found that match the TOI dispositions.")
        print(f"No observations found for {observations_not_found} entries in TOI dispositions.")
        suspected_planet_data_products = tess_data_interface.get_product_list(suspected_planet_observations)
        suspected_planet_lightcurve_data_products = suspected_planet_data_products[
            suspected_planet_data_products['productFilename'].str.endswith('lc.fits')
        ]
        suspected_planet_download_manifest = tess_data_interface.download_products(
            suspected_planet_lightcurve_data_products, data_directory=self.data_directory)
        print(f'Moving lightcurves to {self.synthetic_signal_directory}...')
        for file_path_string in suspected_planet_download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.synthetic_signal_directory.joinpath(file_path.name))

    def create_data_directories(self):
        """
        Creates the data directories to be used by the database.
        """
        super().create_data_directories()
        self.lightcurve_directory.mkdir(parents=True, exist_ok=True)
        self.synthetic_signal_directory.mkdir(parents=True, exist_ok=True)

    def get_all_lightcurve_paths(self) -> Iterable[Path]:
        """
        Returns the list of all lightcurves to use.

        :return: The list of lightcurves.
        """
        lightcurve_paths = self.tess_ffi_data_interface.glob_pickle_path_for_magnitude(self.lightcurve_directory, 9)
        # lightcurve_paths = self.lightcurve_directory.glob('**/*.pkl')
        # lightcurve_paths = self.tess_ffi_data_interface.create_subdirectories_pickle_repeating_generator(
        #     self.lightcurve_directory)
        return lightcurve_paths

    def get_all_synthetic_signal_paths(self) -> Iterable[Path]:
        """
        Returns the list of all synthetic signals to use.

        :return: The list of synthetic signals.
        """
        synthetic_signal_paths = self.synthetic_signal_directory.glob('**/*.fits')
        return synthetic_signal_paths

    def load_fluxes_and_times_from_lightcurve_path(self, lightcurve_path: str) -> (np.ndarray, np.ndarray):
        """
        Loads the lightcurve from the path given. Should be overridden to fit a specific database's file format.

        :param lightcurve_path: The path to the lightcurve file.
        :return: The fluxes and times of the lightcurve
        """
        fluxes, times = self.tess_ffi_data_interface.load_fluxes_and_times_from_pickle_file(lightcurve_path)
        nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.argwhere(np.isnan(times)))
        fluxes = np.delete(fluxes, nan_indexes)
        times = np.delete(times, nan_indexes)
        return fluxes, times

    def load_magnifications_and_times_from_synthetic_signal_path(self, synthetic_signal_path: str
                                                                 ) -> (np.ndarray, np.ndarray):
        """
        Loads the synthetic signal from the path given. Should be overridden to fit a specific database's file format.

        :param synthetic_signal_path: The path to the synthetic signal data file.
        :return: The magnifications and relative times of the synthetic signal.
        """
        if synthetic_signal_path.endswith('.pkl'):
            fluxes, times = self.tess_ffi_data_interface.load_fluxes_and_times_from_pickle_file(synthetic_signal_path)
        else:
            fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return synthetic_magnifications, synthetic_times

    def load_magnifications_and_times_from_negative_synthetic_signal_path(self, synthetic_signal_path: str
                                                                 ) -> (np.ndarray, np.ndarray):
        fluxes, times = self.tess_ffi_data_interface.load_fluxes_and_times_from_pickle_file(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return synthetic_magnifications, synthetic_times

    def get_explicit_negative_synthetic_signal_paths(self) -> Iterable[Path]:
        explicit_negative_lightcurve_paths = list(self.data_directory.joinpath('explicit_negative').glob('**/*.fits'))
        synthetic_signal_names = [path.name for path in self.synthetic_signal_directory.glob('**/*.fits')]
        explicit_negative_lightcurve_paths = [path for path in explicit_negative_lightcurve_paths if path.name
                                              not in synthetic_signal_names]
        hand_selected_negative_data_frame = pd.read_csv(self.data_directory.joinpath('explicit_negatives.csv'))
        hand_selected_negative_paths = list(hand_selected_negative_data_frame['Lightcurve path'].values)
        explicit_negative_lightcurve_paths += hand_selected_negative_paths
        return explicit_negative_lightcurve_paths


if __name__ == '__main__':
    ffi_toi_database = FfiToiDatabase()
    ffi_toi_database.download_exofop_toi_lightcurves_to_synthetic_directory()
