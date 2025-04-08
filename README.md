# Flow:
**Start** -> **Parse Pattern** -> **Input Analysis** -> **Decomposition** -> **Permutation** -> **Transpose** -> **Repetition Check** -> **Broadcast (if needed)** -> **Composition** -> **Validation & Output**

# My Learning Outcome

**Pattern Parsing**: I learned the necessity of rigorously parsing the input string format (e.g., 'b (h w) ... -> ... b h w'). This involved developing logic to identify and differentiate named axes (b, h), composite groups ((h w)) requiring splitting or merging, the ellipsis (...) for arbitrary dimensions, and anonymous axes

**Tensor Shape Decomposition**: A crucial insight was the strategy of mapping the input pattern to the actual input tensor's shape. This often required an initial reshape to bring the tensor into an intermediate "decomposed" state.

**Comparing Input and Output Structures**: I learned to systematically compare the parsed components of the input and output patterns. This comparison is vital for determining the exact transformation required: identifying which axes are retained, reordered, merged, split, or newly introduced through repetition.

**Matching Output to Decomposition via Permutation**: By comparing the desired output axis order with the current order in the decomposed tensor, I could calculate the necessary permutation indices to feed into np.transpose. This step directly links the desired output structure to the intermediate representation.

---

## Usage Considerations: einops vs. Native NumPy

The existence of libraries like einops, and the exercise of recreating its core functionality, prompts consideration of when to use such pattern-based approaches versus standard NumPy functions:

### Use einops (or similar pattern-based functions) when:

**Complex Operations**: You need to combine multiple actions like transposing, reshaping, splitting, and merging dimensions simultaneously.

**Readability and Intent**: The symbolic notation can make the purpose of the tensor manipulation immediately clear, improving code maintainability.

**Reducing Errors**: Explicitly naming axes and defining the transformation in one place can help prevent common off-by-one errors.

### Use Native NumPy Functions (np.reshape, np.transpose, np.expand_dims, etc.) when:

**Simple Operations**: For basic tasks like transposing two axes (tensor.T), adding a dimension (np.expand_dims), or a simple reshape, native functions.

**Performance on Micro-benchmarks**: For extremely performance-critical code involving very simple manipulations on potentially small tensors, a direct NumPy call might be ***fractionally faster***.


---

### Time Complexity Differences:

The primary difference lies in the overhead before these operations execute. einops or a custom implementation like the one discussed involves:

1. Parsing the pattern string.

2. Executing Python logic to determine the sequence of NumPy operations, target shapes, and axis permutations.

3. Native NumPy calls skip this parsing and logic overhead.

Impact: For tensors with a large number of elements, the time spent in the core NumPy operations usually dominates, making the overhead of the pattern-based approach negligible. However, if applying very simple patterns to very small tensors repeatedly in a tight loop, the parsing/logic overhead could become measurable compared to a single direct native NumPy call.


---

# NumPy `rearrange` Implementation

This project provides a Python function `rearrange` that mimics the core functionality of the popular `einops` library's rearrange operation, using only NumPy. It allows flexible tensor manipulation using an intuitive string-based pattern.

## Features Supported

*   **Reshaping:** Changing the number of dimensions and their sizes.
*   **Transposition:** Permuting existing axes.
*   **Splitting Axes:** Dividing one dimension into multiple (e.g., `batch (h w) -> batch h w`).
*   **Merging Axes:** Combining multiple dimensions into one (e.g., `batch h w c -> batch (h w) c`).
*   **Repeating Axes:** Repeating a dimension of size 1 (e.g., `batch 1 c -> batch repeat c`).
*   **Ellipsis (`...`)**: Handling an arbitrary number of leading or trailing dimensions, often used for batch dimensions.


---


## Approach: How it Works Step-by-Step

The `rearrange` function works in several stages to transform the input tensor according to the pattern:

1.  **Pattern Parsing:**
    *   The input pattern string (e.g., `'b (h w) c -> b h w c'`) is split into two parts: the input description (`b (h w) c`) and the output description (`b h w c`).
    *   Each part is parsed to identify its components:
        *   **Named Axes:** Simple identifiers like `b`, `h`, `w`, `c`.
        *   **Anonymous Axes:** The literal number `1` used as a placeholder for dimensions of size 1, typically for repetition.
        *   **Groups:** Axes combined within parentheses like `(h w)`, indicating they correspond to a single dimension in the current tensor shape but should be treated logically as multiple axes (for splitting) or will be combined into one (for merging).
        *   **Ellipsis:** The `...` symbol, representing any number of dimensions not explicitly mentioned.
    *   Basic syntax checks are performed (e.g., valid characters, no duplicate names on one side, matching ellipsis).

2.  **Input Analysis & Validation:**
    *   The parsed *input* description is compared against the actual shape of the input `tensor`.
    *   The size of each named axis (`h`, `w`, etc.) is determined from the tensor's shape.
    *   For **split** groups on the input side (like `(h w)`), the function checks `axes_lengths` provided by the user (e.g., `h=3`). If one part of the group is missing a length, its size is inferred (e.g., if the dimension is 12 and `h=3`, `w` must be 4). An error occurs if sizes don't match the tensor dimension or cannot be inferred.
    *   For `1` axes, it verifies the corresponding tensor dimension actually has size 1.
    *   Ellipsis (`...`) dimensions are calculated by comparing the tensor's rank with the number of explicitly defined axes/groups in the pattern.

3.  **Decomposition (Intermediate Reshape):**
    *   The function performs an initial `reshape` on the input tensor.
    *   The goal is to create an intermediate tensor where *every* logical axis identified in the input pattern (including those originally inside split groups) becomes a separate dimension.
    *   *Example:* If input is `(12, 10)` and pattern is `'(h w) c'` with `h=3, w=4`, the intermediate shape becomes `(3, 4, 10)`.

4.  **Permutation (Transpose):**
    *   The function determines the desired order of axes based on the parsed *output* description.
    *   It calculates the correct permutation (a tuple of axis indices) needed to reorder the axes of the *intermediate (decomposed)* tensor to match the target output structure.
    *   `np.transpose()` is called with this permutation.

5.  **Repetition (Broadcast):**
    *   The function checks if any axes need to be repeated (identified during output parsing where an output axis like `b` corresponds to an input `1` and has a size provided in `axes_lengths`, e.g., `b=4`).
    *   If repetition is needed, it calculates the target shape for NumPy's broadcasting mechanism.
    *   `np.broadcast_to()` is used to create a view of the tensor where the size-1 dimensions have been expanded (repeated) to their target size without copying data unnecessarily.

6.  **Composition (Final Reshape):**
    *   Finally, the tensor (potentially after broadcasting) is reshaped one last time.
    *   This step handles the merging of axes as specified by groups `()` in the *output* pattern.
    *   *Example:* If the pattern is `... -> ... (h w)`, this reshape combines the `h` and `w` dimensions.
    *   The final tensor shape exactly matches the structure defined by the output pattern.

7.  **Output:** The resulting rearranged tensor is returned.



---



## Design Decisions

*   **NumPy Only:** The implementation relies solely on NumPy functions (`reshape`, `transpose`, `broadcast_to`) as requested, avoiding external dependencies like the `einops` library itself.
*   **Intermediate Decomposition:** The initial reshape (decomposition) simplifies the logic significantly. Transposing and repeating axes becomes much easier when each logical component is already a separate dimension.
*   **Explicit Broadcasting:** Using `np.broadcast_to` for repetition is the correct and efficient NumPy approach. Relying only on `reshape` for this would not work as reshape doesn't automatically repeat/broadcast axes.
*   **Clear Error Handling:** The code includes checks at various stages (parsing, input validation, reshaping) and uses a custom `EinopsError` to provide informative messages when patterns or shapes are invalid.
*   **Readability:** The code attempts to follow logical steps corresponding to the conceptual stages of the rearrange operation.
