taxonomy_dict = {
    "": {
        "tags": {
            "cut": {
                "alias": "cropped",
            },
        },
        "children": {
            "Copepoda": {
                "children": {
                    "Calanus": {
                        "children": {
                            "Calanus hyperboreus": {},
                            "Calanus finmarchicus": {},
                        }
                    },
                    "Metridia": {
                        "children": {
                            "Metridia longa": {},
                            "Metridia other": {},
                        },
                    },
                    "Other": {"alias": "*"},
                },
                "tags": {
                    "view": {
                        "children": {
                            "lateral": {
                                "children": {"left": {}, "right": {}},
                            },
                            "frontal": {},
                            "ventral": {},
                        },
                    },
                    "sex": {
                        "children": {
                            "male": {},
                            "female": {},
                        }
                    },
                    "stage": {
                        "children": {
                            "CI": {},
                            "CII": {},
                            "CIII": {},
                            "CIV": {},
                            "CV": {},
                        }
                    },
                },
                "virtuals": {
                    "male+lateral": "male lateral",
                    "male Calanus": "Calanus male",
                    "cropped": "cut",
                    "cvstage": "stage:CV",
                },
            },
        },
    }
}
