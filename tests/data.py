taxonomy_dict = {
    "tags": {
        "cut": {
            "alias": "cropped",
            "meta": {"description": "A cropped image of the organism"},
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
                            "alias": "side",
                            "tags": {"left": {}, "right": {}},
                        },
                        "frontal": {},
                        "ventral": {},
                    },
                },
                "sex": {
                    "tags": {
                        "male": {"alias": "m"},
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
                "male+lateral": "sex:male view:lateral",
                "male Calanus": "Calanus sex:male",
                "cropped": "cut",
                "cvstage": "stage:CV",
            },
        },
    },
}
