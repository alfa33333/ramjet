"""
Code for a database of TESS transit lightcurves with a label per time step.
"""
from pathlib import Path
from typing import List, Dict

from astropy.table import Table
from astroquery.mast import Observations
from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError
from http.client import RemoteDisconnected

from photometric_database.lightcurve_label_per_time_step_database import LightcurveLabelPerTimeStepDatabase


class TessTransitLightcurveLabelPerTimeStepDatabase(LightcurveLabelPerTimeStepDatabase):
    """
    A class for a database of TESS transit lightcurves with a label per time step.
    """
    def __init__(self, data_directory='data/tess'):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.lightcurve_directory = self.data_directory.joinpath('lightcurves')
        self.lightcurve_directory.mkdir(parents=True, exist_ok=True)
        self.data_validation_directory = self.data_directory.joinpath('data_validations')
        self.data_validation_directory.mkdir(parents=True, exist_ok=True)
        self.data_validation_dictionary = None

    def get_lightcurve_file_paths(self) -> List[Path]:
        """
        Gets all the file paths for the available lightcurves.
        """
        return list(self.lightcurve_directory.glob('*.fits'))

    def obtain_data_validation_dictionary(self):
        """
        Collects all the data validation files into a dictionary for fast TIC ID lookup.
        """
        data_validation_dictionary = {}
        for path in self.data_validation_directory.glob('*.xml'):
            tic_id = path.name.split('-')[3]  # The TIC ID is just in the middle of the file name.
            data_validation_dictionary[tic_id] = path
        self.data_validation_dictionary = data_validation_dictionary

    def is_positive(self, example_path: str) -> bool:
        """
        Checks if an example contains a transit event or not.

        :param example_path: The path to the example to check.
        :return: Whether or not the example contains a transit event.
        """
        tic_id = str(Path(example_path).name).split('-')[2]  # The TIC ID is just in the middle of the file name.
        return tic_id in self.data_validation_dictionary

    def download_database(self):
        """
        Downloads the full lightcurve transit database. This includes the lightcurve files and the data validation files
        (which contain the planet threshold crossing event information).
        """
        print('Downloading TESS observation list...')
        tess_observations = None
        while tess_observations is None:
            try:
                tess_observations = Observations.query_criteria(obs_collection='TESS',
                                                                calib_level=3)  # Science data product level.U
            except (AstroQueryTimeoutError, RemoteDisconnected):
                print('Error connecting to MAST. They have occasional downtime. Trying again...')
        for tess_observation in tess_observations:
            download_manifest = None
            while download_manifest is None:
                try:
                    print(f'Downloading data for TIC {tess_observation["target_name"]}...')
                    observation_data_products = Observations.get_product_list(tess_observation)
                    observation_data_products = observation_data_products.to_pandas()
                    lightcurve_and_data_validation_products = observation_data_products[
                        observation_data_products['productFilename'].str.endswith('lc.fits') |
                        observation_data_products['productFilename'].str.endswith('dvr.xml')
                    ]
                    if lightcurve_and_data_validation_products.shape[0] == 0:
                        break  # The observation does not have LC or DVR science products yet.
                    lightcurve_and_data_validation_products = Table.from_pandas(lightcurve_and_data_validation_products)
                    download_manifest = Observations.download_products(lightcurve_and_data_validation_products,
                                                                       download_dir=str(self.data_directory.absolute()))
                    for file_path_string in download_manifest['Local Path']:
                        if file_path_string.endswith('lc.fits'):
                            type_directory = self.lightcurve_directory
                        else:
                            type_directory = self.data_validation_directory
                        file_path = Path(file_path_string)
                        file_path.rename(type_directory.joinpath(file_path.name))
                except (AstroQueryTimeoutError, RemoteDisconnected):
                    print('Error connecting to MAST. They have occasional downtime. Trying again...')


if __name__ == '__main__':
    TessTransitLightcurveLabelPerTimeStepDatabase().download_database()
