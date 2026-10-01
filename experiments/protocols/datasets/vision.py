"""Image benchmark label spaces and pinned Hub image distributions."""
IMAGE_TASKS = {'cifar10': 10, 'cifar100': 100, 'oxford_pets': 37, 'stanford_cars': 196,
               'fgvc_aircraft': 100, 'eurosat': 10, 'resisc45': 45}
IMAGE_HUB = {
    'resisc45': ('timm/resisc45', 'fe12fc5f1b7606543b0355eda392f1ddc54625c6'),
    'stanford_cars': ('tanganke/stanford_cars', '9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40'),
}
