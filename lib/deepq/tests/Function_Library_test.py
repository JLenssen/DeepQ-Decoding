from deepq.Function_Library import multiplyPaulis


class TestFunctionLibrary:

  def test_multiply_paulis(self):
    """
    Test if two pauli operators are correctly multiplied
    """
    pauli_multiplication_table = [
      [0, 1, 2, 3],
      [1, 0, 3, 2],
      [2, 3, 0, 1],
      [3, 2, 1, 0]
    ]

    for i in range(4):
      for j in range(4):
        res = multiplyPaulis(i,j)
        assert res == pauli_multiplication_table[i][j]