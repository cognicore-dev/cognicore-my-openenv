---
name: forget
description: Delete a specific memory from CogniCore by its ID.
usage: /forget <memory-id>
examples:
  - /forget 7
  - /forget 42
---

Delete the memory with the given ID using `cognicore_forget`.

First confirm with the user what memory they are about to delete by calling `cognicore_list` and finding the matching entry. Then call `cognicore_forget` with the ID. Confirm deletion or report if not found.
