import sqlite3
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from unittest.mock import patch, Mock

import ramjet.data_interface.tess_ffi_data_interface
from ramjet.data_interface.tess_ffi_data_interface import TessFfiDataInterface


class TestTessFfiDataInterface:
    @pytest.fixture
    def data_interface(self) -> TessFfiDataInterface:
        """
        Sets up the data interfaced for use in a test.

        :return: The data interface.
        """
        data_interface = TessFfiDataInterface(database_path=':memory:')
        return data_interface

    @pytest.fixture
    def ffi_pickle_contents(self) -> Tuple[int, float, float, float,
                                           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a mock content of one of Brian Powell's FFI data files.

        :return: TIC ID, right ascension, declination, TESS magnitude,
                 time, raw flux, corrected flux, PCA flux, flux error.
        """
        tic_id = 231663901
        ra = 62.2
        dec = -71.4
        tess_magnitude = 10
        time = np.arange(0, 100, 10)
        raw_flux = np.arange(10)
        corrected_flux = np.arange(10, 20)
        pca_flux = np.arange(20, 30)
        flux_error = np.arange(0, 1, 0.1)
        return tic_id, ra, dec, tess_magnitude, time, raw_flux, corrected_flux, pca_flux, flux_error

    @patch.object(ramjet.data_interface.tess_ffi_data_interface.pickle, 'load')
    @patch.object(Path, 'open')
    def test_can_load_flux_and_data_from_ffi_pickle_files(self, mock_open, mock_pickle_load, data_interface,
                                                          ffi_pickle_contents):
        mock_pickle_load.return_value = ffi_pickle_contents
        fake_file_path = Path('fake_path.pkl')
        fluxes, times = data_interface.load_fluxes_and_times_from_pickle_file(fake_file_path)
        assert mock_open.called
        assert np.array_equal(fluxes, ffi_pickle_contents[6])
        assert np.array_equal(times, ffi_pickle_contents[4])

    def test_can_obtain_ffi_pickle_directories(self, data_interface):
        ffi_root_directory = Path('tests/data_interface/test_tess_ffi_data_interface_resources/ffi_lightcurves')
        expected_directories = [
            ffi_root_directory.joinpath('tesslcs_sector_1/tesslcs_tmag_5_6'),
            ffi_root_directory.joinpath('tesslcs_sector_1/tesslcs_tmag_12_13'),
            ffi_root_directory.joinpath('tesslcs_sector_22/tesslcs_tmag_12_13')
        ]
        pickle_directories = data_interface.get_pickle_directories(ffi_root_directory)
        assert sorted(pickle_directories) == sorted(expected_directories)

    def test_can_create_repeating_pickle_glob_dictionary_generator(self, data_interface):
        class MockPath(type(Path())):
            def glob(self, pattern):
                if self.name == 'a':
                    return (item for item in ['1.pkl', '2.pkl'])
                elif self.name == 'b':
                    return (item for item in ['3.pkl'])
                else:
                    return (item for item in [])

        path_a = MockPath('a')
        path_b = MockPath('b')
        path_c = MockPath('c')
        paths = [path_a, path_b, path_c]
        generator = data_interface.create_path_list_pickle_repeating_generator(paths)
        results = []
        for _ in range(8):
            results.append(next(generator))
        assert sorted(results) == ['1.pkl', '1.pkl', '2.pkl', '2.pkl', '3.pkl', '3.pkl', '3.pkl', '3.pkl']

    def test_can_glob_lightcurves_by_magnitude(self, data_interface):
        ffi_root_directory = Path('tests/data_interface/test_tess_ffi_data_interface_resources/ffi_lightcurves')
        expected_paths = [
            ffi_root_directory.joinpath('tesslcs_sector_1/tesslcs_tmag_12_13/fake0.pkl'),
            ffi_root_directory.joinpath('tesslcs_sector_22/tesslcs_tmag_12_13/fake0.pkl')
        ]
        magnitude_filtered_paths = list(data_interface.glob_pickle_path_for_magnitude(ffi_root_directory, 12))
        assert sorted(magnitude_filtered_paths) == sorted(expected_paths)


    @patch.object(ramjet.data_interface.tess_ffi_data_interface.pickle, 'load')
    @patch.object(Path, 'open')
    def test_can_load_fluxes_flux_errors_and_times_from_ffi_pickle_files(self, mock_open, mock_pickle_load,
                                                                         data_interface, ffi_pickle_contents):
        mock_pickle_load.return_value = ffi_pickle_contents
        fake_file_path = Path('fake_path.pkl')
        fluxes, flux_errors, times = data_interface.load_fluxes_flux_errors_and_times_from_pickle_file(fake_file_path)
        assert mock_open.called
        assert np.array_equal(fluxes, ffi_pickle_contents[6])
        assert np.array_equal(flux_errors, ffi_pickle_contents[8])
        assert np.array_equal(times, ffi_pickle_contents[4])

    def test_can_get_tic_id_and_sector_from_ffi_style_file_path(self, data_interface):
        tic_id0, sector0 = data_interface.get_tic_id_and_sector_from_file_path(
            'tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453')
        assert tic_id0 == 290374453
        assert sector0 == 12
        tic_id1, sector1 = data_interface.get_tic_id_and_sector_from_file_path(
            'data/ffi_microlensing_database/lightcurves/tesslcs_sector_1/tesslcs_tmag_12_13/tesslc_1234567.pkl')
        assert tic_id1 == 1234567
        assert sector1 == 1
        tic_id2, sector2 = data_interface.get_tic_id_and_sector_from_file_path('tesslc_12345678.pkl')
        assert tic_id2 == 12345678
        assert sector2 is None

    @patch.object(ramjet.data_interface.tess_ffi_data_interface.sqlite3, 'connect')
    def test_has_a_path_to_lightcurves_directory_with_default(self, mock_connect):
        mock_connect.return_value = Mock(cursor=Mock())
        data_interface0 = TessFfiDataInterface()
        assert data_interface0.lightcurve_root_directory_path == Path('data/tess_ffi_lightcurves')
        data_interface0 = TessFfiDataInterface(lightcurve_root_directory_path=Path('specified/path'))
        assert data_interface0.lightcurve_root_directory_path == Path('specified/path')

    @patch.object(ramjet.data_interface.tess_ffi_data_interface.sqlite3, 'connect')
    def test_has_a_path_to_database_organization_with_default(self, mock_connect):
        mock_connect.return_value = Mock(cursor=Mock())
        data_interface0 = TessFfiDataInterface()
        assert data_interface0.database_path == Path('data/tess_ffi_database.sqlite3')
        data_interface0 = TessFfiDataInterface(database_path=Path('specified/path.sqlite3'))
        assert data_interface0.database_path == Path('specified/path.sqlite3')

    def test_creation_of_database_lightcurve_table_contains_important_columns(self, data_interface):
        data_interface.create_database_lightcurve_table()
        data_interface.database_cursor.execute('select * from Lightcurve')
        column_names = [description[0] for description in data_interface.database_cursor.description]
        assert 'path' in column_names
        assert 'magnitude' in column_names
        assert 'dataset_split' in column_names

    def test_can_get_floor_magnitude_from_ffi_style_file_path(self, data_interface):
        magnitude0 = data_interface.get_floor_magnitude_from_file_path(
            'tesslcs_sector_12/tesslcs_tmag_2_3/tesslc_290374453')
        assert magnitude0 == 2
        magnitude1 = data_interface.get_floor_magnitude_from_file_path(
            'data/ffi_microlensing_database/lightcurves/tesslcs_sector_1/tesslcs_tmag_14_15/tesslc_1234567.pkl')
        assert magnitude1 == 14
        with pytest.raises(ValueError):
            data_interface.get_floor_magnitude_from_file_path('tesslc_12345678.pkl')

    def test_can_add_sql_database_lightcurve_row_from_path(self, data_interface):
        data_interface.create_database_lightcurve_table()
        lightcurve_path0 = Path('tesslcs_sector_1/tesslcs_tmag_7_8/tesslc_1111.pkl')
        uuid0 = 'mock-uuid-output0'
        with patch.object(ramjet.data_interface.tess_ffi_data_interface, 'uuid4') as mock_uuid4:
            mock_uuid4.return_value = uuid0
            data_interface.insert_database_lightcurve_row_from_path(lightcurve_path=lightcurve_path0, dataset_split=2)
        lightcurve_path1 = Path('tesslcs_sector_1/tesslcs_tmag_14_15/tesslc_1234567.pkl')
        uuid1 = 'mock-uuid-output1'
        with patch.object(ramjet.data_interface.tess_ffi_data_interface, 'uuid4') as mock_uuid4:
            mock_uuid4.return_value = uuid1
            data_interface.insert_database_lightcurve_row_from_path(lightcurve_path=lightcurve_path1, dataset_split=3)
        data_interface.database_cursor.execute('SELECT uuid, path, magnitude, dataset_split FROM Lightcurve')
        query_result = data_interface.database_cursor.fetchall()
        assert query_result == [(uuid0, str(lightcurve_path0), 7, 2),
                                (uuid1, str(lightcurve_path1), 14, 3)]

    def test_uuid_is_primary_key_of_sql_database(self, data_interface):
        data_interface.create_database_lightcurve_table()
        # noinspection SqlResolve
        data_interface.database_cursor.execute('SELECT name FROM pragma_table_info("Lightcurve") WHERE pk=1')
        first_query_result = data_interface.database_cursor.fetchone()
        assert first_query_result[0] == 'uuid'

    def test_indexes_of_sql_database(self, data_interface):
        data_interface.create_database_lightcurve_table()
        # noinspection SqlResolve
        results_data_frame = pd.read_sql_query('''SELECT index_list.seq AS index_sequence,
                                                         seqno as index_sequence_number,
                                                         index_info.name as column_name
                                                  FROM pragma_index_list("Lightcurve") index_list,
                                                       pragma_index_info(index_list.name) index_info;''',
                                               data_interface.database_connection)
        sorted_index_groups = results_data_frame.sort_values('index_sequence_number').groupby('index_sequence')
        column_lists_of_indexes = list(sorted_index_groups['column_name'].apply(list).values)
        assert ['magnitude', 'dataset_split', 'uuid'] in column_lists_of_indexes
        assert ['dataset_split', 'uuid'] in column_lists_of_indexes
        assert ['uuid'] in column_lists_of_indexes

    @patch.object(Path, 'glob')
    def test_can_populate_sql_dataset_from_ffi_directory(self, mock_glob, data_interface):
        data_interface.get_floor_magnitude_from_file_path = Mock(return_value=0)
        data_interface.create_database_lightcurve_table()
        path_list = [Path(f'{index}.pkl') for index in range(20)]
        mock_glob.return_value = path_list
        data_interface.populate_sql_database()
        results_data_frame = pd.read_sql_query('SELECT path, dataset_split FROM Lightcurve',
                                               data_interface.database_connection)
        dataset_split_sizes = results_data_frame.groupby('dataset_split').size()
        assert len(dataset_split_sizes) == 10
        assert all(dataset_split_sizes.values == 2)
        assert sorted(list(map(str, path_list))) == sorted(list(results_data_frame['path'].values))

    def test_unique_columns_of_sql_table(self, data_interface):
        data_interface.create_database_lightcurve_table()
        # noinspection SqlResolve
        results_data_frame = pd.read_sql_query('''SELECT index_list.seq AS index_sequence,
                                                         seqno as index_sequence_number,
                                                         index_info.name as column_name
                                                  FROM pragma_index_list("Lightcurve") index_list, 
                                                       pragma_index_info(index_list.name) index_info
                                                  WHERE index_list.origin='u';''',
                                               data_interface.database_connection)
        sorted_index_groups = results_data_frame.sort_values('index_sequence_number').groupby('index_sequence')
        column_lists_of_unique_indexes = list(sorted_index_groups['column_name'].apply(list).values)
        assert ['path'] in column_lists_of_unique_indexes
