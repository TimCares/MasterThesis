from imagenet_zeroshot_data import imagenet_classnames

prefixes = [
    "a photo of a", 
    "a good photo of a", 
    "a bad photo of a", 
    "a closeup photo of a", 
    "itap of a",
]

suffixes = [
    "I like it.",
    "It's common in daily life.",
    "It's not common in daily life.",
    "It's ugly.",
    "It's cute.",
    "It's beautiful."
]

filip_prompt_templates = [
    f"{prefix} {classname}. {suffix}"
    for classname in imagenet_classnames
    for prefix in prefixes
    for suffix in suffixes
]
