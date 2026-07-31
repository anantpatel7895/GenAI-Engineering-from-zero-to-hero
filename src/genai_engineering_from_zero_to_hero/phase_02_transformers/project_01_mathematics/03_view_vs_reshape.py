"""
Lesson 03: view() vs reshape()

This example demonstrates:
1. Contiguous vs non-contiguous tensors
2. Why view() requires contiguous memory
3. Why reshape() works on non-contiguous tensors
4. Memory sharing using data_ptr()
"""

import torch


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    """Print useful tensor information."""

    print(f"\n{name}")
    print("-" * 50)
    print(f"Shape         : {tuple(tensor.shape)}")
    print(f"Stride        : {tensor.stride()}")
    print(f"Contiguous    : {tensor.is_contiguous()}")
    print(f"Data Pointer  : {tensor.data_ptr()}")
    print(tensor)


def main() -> None:
    print("=" * 70)
    print("Step 1 : Create a contiguous tensor")
    print("=" * 70)

    x = torch.arange(12)

    describe_tensor("x", x)

    print("\nReshape using view()")

    y = x.view(3, 4)

    describe_tensor("y = x.view(3, 4)", y)

    print("\nMemory shared?")
    print(f"x.data_ptr() == y.data_ptr() : {x.data_ptr() == y.data_ptr()}")

    print("\n" + "=" * 70)
    print("Step 2 : Transpose the tensor")
    print("=" * 70)

    z = y.T

    describe_tensor("z = y.T", z)

    print("\nMemory shared?")
    print(f"y.data_ptr() == z.data_ptr() : {y.data_ptr() == z.data_ptr()}")

    print("\n" + "=" * 70)
    print("Step 3 : Attempt view()")
    print("=" * 70)

    try:
        z_view = z.view(12)
        describe_tensor("z.view(12)", z_view)

    except RuntimeError as error:
        print("\nview() failed!")
        print(error)

    print("\n" + "=" * 70)
    print("Step 4 : Use reshape()")
    print("=" * 70)

    z_reshape = z.reshape(12)

    describe_tensor("z.reshape(12)", z_reshape)

    print("\nMemory shared?")
    print(f"z.data_ptr() == z_reshape.data_ptr() : {z.data_ptr() == z_reshape.data_ptr()}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("x and y share memory :", x.data_ptr() == y.data_ptr())
    print("y and z share memory :", y.data_ptr() == z.data_ptr())
    print("z and reshape share memory :", z.data_ptr() == z_reshape.data_ptr())

    print("\nConclusion:")
    print("- view() only changes metadata.")
    print("- view() requires contiguous memory.")
    print("- transpose() changes only metadata and produces a non-contiguous tensor.")
    print("- reshape() allocates new memory when required.")
    print("- reshape() returned a different data pointer because a new tensor was created.")


if __name__ == "__main__":
    main()

"""
nterview Question

After running this program, answer these questions without looking at the notes:

Why do x, y, and z have the same data_ptr()?
Why is z non-contiguous even though it shares the same memory?
Why does view() fail on z?
Why does reshape() have a different data_ptr() than z?
"""