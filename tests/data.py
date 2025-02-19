taxonomy_dict = {
    "tags": {
        "cut": {
            "alias": "cropped",
        },
    },
    "classes": {
        "Copepoda": {
            "classes": {
                "Calanus": {
                    "classes": {
                        "Calanus hyperboreus": {},
                        "Calanus finmarchicus": {},
                    }
                },
                "Metridia": {
                    "classes": {
                        "Metridia longa": {},
                        "Metridia other": {},
                    },
                },
                "Other": {"alias": "*"},
            },
            "tags": {
                "view": {
                    "tags": {
                        "lateral": {
                            "tags": {"left": {}, "right": {}},
                        },
                        "frontal": {},
                        "ventral": {},
                    },
                },
                "sex": {
                    "tags": {
                        "male": {},
                        "female": {},
                    }
                },
                "stage": {
                    "tags": {
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
