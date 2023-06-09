from dataclasses import dataclass
from time import perf_counter
from typing import TypedDict

import torch
from torch.utils.data import Dataset, DataLoader, default_collate


# Simple One Level

@dataclass(slots=True)
class DSSimpleItem:
    item: torch.Tensor

def simple_collate(batch):
    return DSSimpleItem(item=default_collate([x.item for x in batch]))

class DSSimple(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return DSSimpleItem(item=self.items[idx])
    
def extract_dssimple(batch):
    return batch.item
    

class TDSimpleItem(TypedDict):
    item: torch.Tensor

class TDSimple(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return TDSimpleItem(item=self.items[idx])
    
def extract_tdsimple(batch):
    return batch["item"]
    

class DSimple(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return dict(item=self.items[idx])
    
def extract_dsimple(batch):
    return batch["item"]
    

class TSimple(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
def extract_tsimple(batch):
    return batch
    

# Nested (Three Levels)

@dataclass(slots=True)
class DSNestedSubSub:
    item1: torch.Tensor
    item2: torch.Tensor

@dataclass(slots=True)
class DSNestedSub:
    item: torch.Tensor
    sub: DSNestedSubSub

@dataclass(slots=True)
class DSNestedItem:
    item: torch.Tensor
    sub: DSNestedSub

def nested_collate(batch):
    return DSNestedItem(
        item=default_collate([x.item for x in batch]),
        sub=DSNestedSub(
            item=default_collate([x.sub.item for x in batch]),
            sub=DSNestedSubSub(
                item1=default_collate([x.sub.sub.item1 for x in batch]),
                item2=default_collate([x.sub.sub.item2 for x in batch])
            )
        )
    )

class DSNested(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return DSNestedItem(
            item=self.items[idx],
            sub=DSNestedSub(
                item=self.items[idx]**2,
                sub=DSNestedSubSub(
                    item1=self.items[idx]**2 - 3.7,
                    item2=1/(self.items[idx]+1e-4)
                )
            )
        )
    
def extract_dsnested(batch):
    return batch.item, batch.sub.item, batch.sub.sub.item1, batch.sub.sub.item2
    

class TDNestedSubSub(TypedDict):
    item1: torch.Tensor
    item2: torch.Tensor

class TDNestedSub(TypedDict):
    item: torch.Tensor
    sub: TDNestedSubSub

class TDNestedItem(TypedDict):
    item: torch.Tensor
    sub: TDNestedSub

class TDNested(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return TDNestedItem(
            item=self.items[idx],
            sub=TDNestedSub(
                item=self.items[idx]**2,
                sub=TDNestedSubSub(
                    item1=self.items[idx]**2 - 3.7,
                    item2=1/(self.items[idx]+1e-4)
                )
            )
        )
    
def extract_tdnested(batch):
    return batch["item"], batch["sub"]["item"], batch["sub"]["sub"]["item1"], batch["sub"]["sub"]["item2"]
    

class DNested(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return dict(
            item=self.items[idx],
            sub=dict(
                item=self.items[idx]**2,
                sub=dict(
                    item1=self.items[idx]**2 - 3.7,
                    item2=1/(self.items[idx]+1e-4)
                )
            )
        )
    
def extract_dnested(batch):
    return batch["item"], batch["sub"]["item"], batch["sub"]["sub"]["item1"], batch["sub"]["sub"]["item2"]


class TNested(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        sub_item = item**2
        sub_sub_item1 = item**2 - 3.7
        sub_sub_item2 = 1/(item+1e-4)
        return torch.concat((item, sub_item, sub_sub_item1, sub_sub_item2))
    
def extract_tnested(batch):
    return batch[:,0], batch[:,1], batch[:,2], batch[:,3]

## Benchmark

def output(cls, inp, name, collate_fn):
    print(name)
    ds = cls(inp)
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    for batch in dl:
        print(batch)
        break

def show_example_output():
    print("SIMPLE")
    data = torch.tensor([1,2,3], dtype=torch.float32)
    output(DSSimple, data, "dataclass", simple_collate)
    output(TDSimple, data, "TypedDict", default_collate)
    output(DSimple, data, "dict", default_collate)
    output(TSimple, data, "tensor", default_collate)
    
    print("\nNESTED")
    data = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32)
    output(DSNested, data, "dataclass", nested_collate)
    output(TDNested, data, "TypedDict", default_collate)
    output(DNested, data, "dict", default_collate)
    output(TNested, data, "tensor", default_collate)


def benchmark_one(cls, inp, name, collate_fn, tries, extract_fn):
    tdata = 0
    titer = 0
    textr = 0
    st = perf_counter()
    for _ in range(tries):
        st_data = perf_counter()
        ds = cls(inp)
        dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        tdata += perf_counter() - st_data
        st_iter = perf_counter()
        for batch in dl:
            st_extr = perf_counter()
            extracted = extract_fn(batch)
            textr += perf_counter() - st_extr
        titer += perf_counter() - st_iter
    total = perf_counter() - st
    print(f"{name} results")
    print(f"total: {total:.3f}s, {total/tries:.3f}s/epoch")
    print(f"  iteration: {titer:.3f}s, {titer/tries:.3f}s/epoch")
    print(f"    extraction: {textr:.3f}s, {textr/tries:.3f}s/epoch")

def benchmark():
    TRIES = 10
    batch_size = 256
    size = batch_size * 100
    print(f"Benchmarking with {size} items, {batch_size} batch size, {TRIES} tries")

    print("\nSIMPLE")
    data = torch.rand(size, dtype=torch.float32)
    benchmark_one(DSSimple, data, "dataclass", simple_collate, TRIES, extract_dssimple)
    benchmark_one(TDSimple, data, "TypedDict", default_collate, TRIES, extract_tdsimple)
    benchmark_one(DSimple, data, "dict", default_collate, TRIES, extract_dsimple)
    benchmark_one(TSimple, data, "tensor", default_collate, TRIES, extract_tsimple)

    print("\nNESTED")
    data = torch.rand(size, 3, dtype=torch.float32)
    benchmark_one(DSNested, data, "dataclass", nested_collate, TRIES, extract_dsnested)
    benchmark_one(TDNested, data, "TypedDict", default_collate, TRIES, extract_tdnested)
    benchmark_one(DNested, data, "dict", default_collate, TRIES, extract_dnested)
    benchmark_one(TNested, data, "tensor", default_collate, TRIES, extract_tnested)


if __name__ == "__main__":
    show_example_output()
    benchmark()


'''
SIMPLE
dataclass
DSSimpleItem(item=tensor([1., 2.]))
TypedDict
{'item': tensor([1., 2.])}
dict
{'item': tensor([1., 2.])}
tensor
tensor([1., 2.])

NESTED
dataclass
DSNestedItem(item=tensor([[1., 2., 3.],
        [4., 5., 6.]]), sub=DSNestedSub(item=tensor([[ 1.,  4.,  9.],
        [16., 25., 36.]]), sub=DSNestedSubSub(item1=tensor([[-2.7000,  0.3000,  5.3000],
        [12.3000, 21.3000, 32.3000]]), item2=tensor([[0.9999, 0.5000, 0.3333],
        [0.2500, 0.2000, 0.1667]]))))
TypedDict
{'item': tensor([[1., 2., 3.],
        [4., 5., 6.]]), 'sub': {'item': tensor([[ 1.,  4.,  9.],
        [16., 25., 36.]]), 'sub': {'item1': tensor([[-2.7000,  0.3000,  5.3000],
        [12.3000, 21.3000, 32.3000]]), 'item2': tensor([[0.9999, 0.5000, 0.3333],
        [0.2500, 0.2000, 0.1667]])}}}
dict
{'item': tensor([[1., 2., 3.],
        [4., 5., 6.]]), 'sub': {'item': tensor([[ 1.,  4.,  9.],
        [16., 25., 36.]]), 'sub': {'item1': tensor([[-2.7000,  0.3000,  5.3000],
        [12.3000, 21.3000, 32.3000]]), 'item2': tensor([[0.9999, 0.5000, 0.3333],
        [0.2500, 0.2000, 0.1667]])}}}
tensor
tensor([[ 1.0000,  2.0000,  3.0000,  1.0000,  4.0000,  9.0000, -2.7000,  0.3000,
          5.3000,  0.9999,  0.5000,  0.3333],
        [ 4.0000,  5.0000,  6.0000, 16.0000, 25.0000, 36.0000, 12.3000, 21.3000,
         32.3000,  0.2500,  0.2000,  0.1667]])
Benchmarking with 25600 items, 256 batch size, 10 tries

SIMPLE
dataclass results
total: 3.140s, 0.314s/epoch
  iteration: 3.140s, 0.314s/epoch
    extraction: 0.063s, 0.006s/epoch
TypedDict results
total: 3.616s, 0.362s/epoch
  iteration: 3.615s, 0.362s/epoch
    extraction: 0.068s, 0.007s/epoch
dict results
total: 3.458s, 0.346s/epoch
  iteration: 3.458s, 0.346s/epoch
    extraction: 0.067s, 0.007s/epoch
tensor results
total: 2.940s, 0.294s/epoch
  iteration: 2.940s, 0.294s/epoch
    extraction: 0.061s, 0.006s/epoch

NESTED
dataclass results
total: 16.033s, 1.603s/epoch
  iteration: 16.032s, 1.603s/epoch
    extraction: 0.210s, 0.021s/epoch
TypedDict results
total: 18.091s, 1.809s/epoch
  iteration: 18.090s, 1.809s/epoch
    extraction: 0.231s, 0.023s/epoch
dict results
total: 16.810s, 1.681s/epoch
  iteration: 16.809s, 1.681s/epoch
    extraction: 0.214s, 0.021s/epoch
tensor results
total: 15.281s, 1.528s/epoch
  iteration: 15.280s, 1.528s/epoch
    extraction: 2.031s, 0.203s/epoch
'''

