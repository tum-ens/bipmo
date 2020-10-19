'''
    Module that contains useful utility methods that are used throughout bipmo
'''

import typing
import pandas as pd


class ResultsDict(typing.Dict[str, pd.DataFrame]):
    """Results dictionary object, i.e., a modified dictionary object with strings as keys and dataframes as values.

    - When printed or represented as string, all dataframes are printed in full.
    - Provides a method for storing all results dataframes to CSV files.
    """

    def __repr__(self) -> str:
        """Obtain string representation of results."""

        repr_string = ""
        for key in self:
            repr_string += f"{key} = \n{self[key]}\n"

        return repr_string

    def to_csv(self, path: str) -> None:
        """Store results to CSV files at given `path`."""

        for key in self:
            self[key].to_csv(os.path.join(path, f'{key}.csv'))
