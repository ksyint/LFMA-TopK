"""CUDA spectral measurements for sparse Fourier adapter checkpoints."""

import argparse
import csv
import json
from pathlib import Path

import torch

from lfma.artifacts.validation import validate_adapter
from lfma.core import BACKBONES, cuda_device


class SpectralInspector:
    def __init__(self, checkpoint, device='cuda'):
        from safetensors.torch import load_file

        self.directory = Path(checkpoint).resolve()
        self.device = cuda_device(device)
        self.audit = validate_adapter(self.directory)
        self.metadata = json.loads((self.directory / 'adapter_config.json').read_text())
        self.width = BACKBONES[self.metadata['config']['model']['backbone']]['width']
        self.tensors = load_file(
            str(self.directory / 'adapter_model.safetensors'), device=str(self.device)
        )
        self.supports = {}
        for name, spec in self.metadata['targets'].items():
            indices = self.tensors[name + '.indices']
            coefficients = self.tensors[name + '.c']
            if (
                indices.unique().numel() != spec['k']
                or indices.min() < 0
                or indices.max() >= self.width**2
            ):
                raise ValueError(f'Invalid support indices at {name}')
            if not torch.isfinite(coefficients).all():
                raise ValueError(f'Nonfinite Fourier coefficients at {name}')
            self.supports[name] = indices

    def coefficients(self, name):
        values = self.tensors[name + '.c'].float().contiguous()
        return torch.view_as_complex(values)

    def grid(self, name):
        values = self.coefficients(name)
        grid = torch.zeros(self.width**2, device=self.device, dtype=torch.complex64)
        return grid.scatter(0, self.supports[name], values).view(self.width, self.width)

    def frequencies(self, name):
        indices = self.supports[name]
        axis = torch.fft.fftfreq(self.width, device=self.device)
        row = torch.div(indices, self.width, rounding_mode='floor')
        column = indices.remainder(self.width)
        return axis[row], axis[column]

    def conjugate_coverage(self, name):
        indices = self.supports[name]
        rows = torch.div(indices, self.width, rounding_mode='floor')
        columns = indices.remainder(self.width)
        partners = (-rows).remainder(self.width) * self.width + (-columns).remainder(self.width)
        membership = torch.isin(partners, indices)
        return {
            'paired_indices': int(membership.sum()),
            'paired_fraction': float(membership.float().mean()),
            'self_conjugate_indices': int((partners == indices).sum()),
        }

    def radial_energy(self, name, bins=8):
        if bins < 1:
            raise ValueError('Radial bin count must be positive')
        row, column = self.frequencies(name)
        radius = torch.sqrt(row.square() + column.square())
        edges = torch.linspace(0, 2**-0.5 + 1e-6, bins + 1, device=self.device)
        buckets = torch.bucketize(radius, edges[1:-1])
        energy = self.coefficients(name).abs().square()
        total = energy.sum()
        rows = []
        for index in range(bins):
            mask = buckets == index
            rows.append(
                {
                    'layer': name,
                    'bin': index,
                    'radius_min': float(edges[index]),
                    'radius_max': float(edges[index + 1]),
                    'support': int(mask.sum()),
                    'energy': float(energy[mask].sum()),
                    'energy_fraction': float(energy[mask].sum() / total) if total > 0 else 0.0,
                }
            )
        return rows

    def layer_summary(self, name):
        values = self.coefficients(name)
        magnitude = values.abs()
        energy = magnitude.square()
        total = energy.sum()
        distribution = energy / total.clamp_min(torch.finfo(torch.float32).tiny)
        entropy = -(distribution * distribution.clamp_min(1e-30).log()).sum()
        spatial = torch.fft.ifft2(self.grid(name))
        scale = self.metadata['targets'][name]['alpha']
        delta = spatial.real * scale
        result = {
            'layer': name,
            'support': len(values),
            'real_parameters': 2 * len(values),
            'support_fraction': len(values) / self.width**2,
            'alpha': scale,
            'coefficient_l2': float(total.sqrt()),
            'coefficient_max': float(magnitude.max()),
            'coefficient_mean': float(magnitude.mean()),
            'energy_entropy': float(entropy),
            'effective_frequencies': float(entropy.exp()) if total > 0 else 0.0,
            'delta_frobenius': float(delta.norm()),
            'delta_abs_max': float(delta.abs().max()),
            'delta_mean': float(delta.mean()),
            'delta_std': float(delta.std(unbiased=False)),
            'imaginary_reconstruction_l2': float(spatial.imag.norm()),
            'dc_selected': bool((self.supports[name] == 0).any()),
            **self.conjugate_coverage(name),
        }
        return result

    def largest_coefficients(self, name, count=20):
        if count < 1:
            raise ValueError('Coefficient listing count must be positive')
        values = self.coefficients(name)
        order = values.abs().argsort(descending=True, stable=True)[:count]
        indices = self.supports[name][order]
        selected = values[order]
        return [
            {
                'layer': name,
                'rank': rank + 1,
                'index': int(index),
                'row': int(index // self.width),
                'column': int(index % self.width),
                'real': float(value.real),
                'imaginary': float(value.imag),
                'magnitude': float(value.abs()),
                'phase': float(value.angle()),
            }
            for rank, (index, value) in enumerate(zip(indices, selected))
        ]

    def compare(self, other):
        if self.metadata['base_model'] != other.metadata['base_model']:
            raise ValueError('Support comparison requires the same base transformer')
        if self.metadata['targets'].keys() != other.metadata['targets'].keys():
            raise ValueError('Support comparison requires identical target layers')
        rows = []
        for name in sorted(self.supports):
            left = self.supports[name]
            right = other.supports[name].to(self.device)
            overlap = int(torch.isin(left, right).sum())
            union = len(left) + len(right) - overlap
            rows.append(
                {
                    'layer': name,
                    'left_support': len(left),
                    'right_support': len(right),
                    'intersection': overlap,
                    'jaccard': overlap / union,
                    'retained_left_fraction': overlap / len(left),
                }
            )
        return rows

    def head_summary(self):
        rows = []
        for name, tensor in sorted(self.tensors.items()):
            if not name.startswith('classifier.'):
                continue
            values = tensor.float()
            if not torch.isfinite(values).all():
                raise ValueError(f'Nonfinite task-head tensor at {name}')
            rows.append(
                {
                    'name': name,
                    'shape': list(values.shape),
                    'parameters': values.numel(),
                    'l2': float(values.norm()),
                    'mean': float(values.mean()),
                    'std': float(values.std(unbiased=False)),
                    'abs_max': float(values.abs().max()),
                }
            )
        return rows

    @torch.no_grad()
    def export(self, output, bins=8, top=20, comparison=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        layers, radial, coefficients = [], [], []
        for name in sorted(self.supports):
            layers.append(self.layer_summary(name))
            radial.extend(self.radial_energy(name, bins))
            coefficients.extend(self.largest_coefficients(name, top))
        tables = {'layers': layers, 'radial_energy': radial, 'coefficients': coefficients}
        if comparison is not None:
            tables['support_overlap'] = self.compare(comparison)
        for name, rows in tables.items():
            with (output / f'{name}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        report = {
            'format': 'lfma-spectrum-v1',
            'checkpoint': str(self.directory),
            'backbone': self.audit['backbone'],
            'task': self.audit['task'],
            'epoch': self.metadata['epoch'],
            'width': self.width,
            'layers': layers,
            'head': self.head_summary(),
            'total_adapter_real_scalars': sum(row['real_parameters'] for row in layers),
        }
        (output / 'spectrum.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n'
        )
        return report


def spectrum_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compare')
    parser.add_argument('--bins', type=int, default=8)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.bins, args.top) < 1:
        parser.error('--bins and --top must be positive')
    inspector = SpectralInspector(args.checkpoint, args.device)
    comparison = SpectralInspector(args.compare, args.device) if args.compare else None
    print(json.dumps(inspector.export(args.output, args.bins, args.top, comparison), indent=2))
