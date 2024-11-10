from typing import Literal

class Feature:
    """
    A class used to represent a Feature in a dataset.

    Attributes
    ----------
    name : str
        The name of the feature.
    type : Literal["numerical", "categorical"]
        The type of the feature, either 'numerical' or 'categorical'.
    """

    def __init__(self, name: str, type: Literal["numerical", "categorical"]) -> None:
        """
        Parameters
        ----------
        name : str
            The name of the feature.
        type : Literal["numerical", "categorical"]
            The type of the feature.
        """
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """
        Gets the name of the feature.

        Returns
        -------
        str
            The name of the feature.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        Sets the name of the feature.

        Parameters
        ----------
        name : str
            The new name of the feature.

        Raises
        ------
        ValueError
            If the name is an empty string.
        """
        if len(name) == 0:
            raise ValueError("Name must be a non empty string")
        if isinstance(name, str):
            self._name = name

    @property
    def type(self) -> Literal["numerical", "categorical"]:
        """
        Gets the type of the feature.

        Returns
        -------
        Literal["numerical", "categorical"]
            The type of the feature.
        """
        return self._type

    @type.setter
    def type(self, type: Literal["numerical", "categorical"]) -> None:
        """
        Sets the type of the feature.

        Parameters
        ----------
        type : Literal["numerical", "categorical"]
            The new type of the feature.

        Raises
        ------
        ValueError
            If the type is not 'numerical' or 'categorical'.
        """
        if type not in {"numerical", "categorical"}:
            raise ValueError(
                f"Type must be either 'numerical' or 'categorical'. "
                f"Received {type}"
            )
        self._type = type

    def __str__(self) -> str:
        """
        Returns a string representation of the feature.

        Returns
        -------
        str
            A string representation of the feature.
        """
        return f"Feature(name={self._name}, type={self._type})"
