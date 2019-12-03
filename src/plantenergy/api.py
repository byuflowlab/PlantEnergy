"""
Authors: Daniel Jolley and Jared Thomas
Date: September 2019
"""
import yaml

import plantenergy.api as pe



# def setup_farm(object_with_a_dict, wind_turbine_definition="some_defaul_file.yaml", wind_farm_definition="some_defaul_file.yaml"):
#
#     # get stuff from files
#     wind_turbine =  yaml.safeload?.load(wind_turbine_definition)
#     wind_farm =  yaml.safeload?(wind_farm_definition)
#
#     # unpack yaml
#
#     # assign the stuff to object_with_a_dict
#     object_with_a_dict['turbineX'] = turbineX # numpy array of size nTurbines
#     object_with_a_dict['turbineY'] = turbineY
#     object_with_a_dict['rotorDiameter'] = rotorDiameter
#     object_with_a_dict['hubHeight'] = hubHeight
#
#     return object_with_a_dict
#
#
# if __name__ == "__main__":
#     object_with_a_dict = ??
#     object_with_a_dict = pe.setup_farm(object_with_a_dict, file1, file2)