class Animal:
    def __init__(self, animal_id, name=None, dob=None, sex='M', type_=None):
        self._id = animal_id
        self.name = name
        self.sex = sex
        self._dob = dob
        self.type_ = type_

    @property
    def id(self):
        """Unique animal id."""
        return self._id

    @property
    def dob(self):
        """Date of birth."""
        return self._dob

class Rat(Animal):
    """See http://www.ratbehavior.org/RatSpecies.htm"""
    def __init__(self, animal_id, *args, name=None, type_='Long Evans', **kwargs):
        # kwargs = {"id": id,
        #           "unit_ids": unit_ids,
        #           "unit_labels": unit_labels,
        #           "unit_tags": unit_tags,
        #           "label": label}

        super().__init__(animal_id=animal_id, *args, name=name, **kwargs)

    def __repr__(self):
        return "<Rat: " + self.name + ">"
