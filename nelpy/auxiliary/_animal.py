class Animal:
    """
    Animal metadata container for experimental subjects.

    Parameters
    ----------
    animal_id : str or int
        Unique identifier for the animal.
    name : str, optional
        Name of the animal.
    dob : str or datetime, optional
        Date of birth.
    sex : str, optional
        Sex of the animal (default 'M').
    type_ : str, optional
        Type or strain of the animal.

    Attributes
    ----------
    id : str or int
        Unique animal id.
    name : str
        Name of the animal.
    dob : str or datetime
        Date of birth.
    sex : str
        Sex of the animal.
    type_ : str
        Type or strain of the animal.
    """
    def __init__(self, animal_id, name=None, dob=None, sex="M", type_=None):
        self._id = animal_id
        self.name = name
        self.sex = sex
        self._dob = dob
        self.type_ = type_

    @property
    def id(self):
        """
        Unique animal id.

        Returns
        -------
        id : str or int
            Unique identifier for the animal.
        """
        return self._id

    @property
    def dob(self):
        """
        Date of birth.

        Returns
        -------
        dob : str or datetime
            Date of birth of the animal.
        """
        return self._dob


class Rat(Animal):
    """
    Rat metadata container, extending Animal.

    See http://www.ratbehavior.org/RatSpecies.htm

    Parameters
    ----------
    animal_id : str or int
        Unique identifier for the rat.
    name : str, optional
        Name of the rat.
    type_ : str, optional
        Strain of the rat (default 'Long Evans').
    *args, **kwargs :
        Additional arguments passed to Animal.
    """
    def __init__(self, animal_id, *args, name=None, type_="Long Evans", **kwargs):
        super().__init__(animal_id=animal_id, *args, name=name, **kwargs)

    def __repr__(self):
        """
        Return a string representation of the Rat.

        Returns
        -------
        repr_str : str
            String representation of the Rat.
        """
        return "<Rat: " + str(self.name) + ">"
