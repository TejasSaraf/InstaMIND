

"""
Single source of truth for incident class definitions and mappings.
Every pipeline script imports from here — never hardcode class names elsewhere.
"""





INSTAMIND_CLASSES = [

    "fighting",

    "robbery",

    "shoplifting",

    "shooting",

    "fainting",

    "normal",

]







UCF_TO_INSTAMIND = {

    "Fighting":            "fighting",



    "Robbery":             "robbery",



    "Shoplifting":         "shoplifting",



    "Shooting":            "shooting",



    "Normal_Videos":       "normal",







    "Burglary":            None,

    "Assault":             None,

    "Abuse":               None,

    "Stealing":            None,

    "Arrest":              None,

    "Arson":               None,

    "Explosion":           None,

    "RoadAccidents":       None,

    "Vandalism":           None,

}





URFD_FOLDER_TO_INSTAMIND = {

    "fall": "fainting",

    "adl":  "normal",

}





INSTAMIND_TO_UCF = {}

for ucf_cls, instamind_cls in UCF_TO_INSTAMIND.items():

    if instamind_cls:

        INSTAMIND_TO_UCF.setdefault(instamind_cls, []).append(ucf_cls)



def ucf_class_of(ucf_folder_name: str) -> str | None:

    """Map a UCF-Crime folder name to instaMIND class. None = excluded."""

    return UCF_TO_INSTAMIND.get(ucf_folder_name)



def urfd_class_of(urfd_folder_name: str) -> str | None:

    """Map a URFD folder name to instaMIND class. None = excluded."""

    return URFD_FOLDER_TO_INSTAMIND.get(urfd_folder_name.lower())