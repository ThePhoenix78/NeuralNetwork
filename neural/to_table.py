from typing import Any, List, Optional
from copy import deepcopy


def make_table(rows: List[List[Any]], labels: Optional[List[Any]] = None, left: Optional[List[Any]] = None, centered: bool = False) -> str:
    full_list: list = deepcopy(rows)

    if labels:
        l: list = labels.copy()
        full_list.insert(0, l)
        del l

    if left:
        l: list = left.copy()
        [full_list[i].insert(0, l[i]) for i in range(len(full_list))]
        del l

    maxi: list = [len(max([str(elem[i]) for elem in full_list], key=len))+2 for i in range(len(full_list[0]))]

    frame: list = ["┌"]
    cut: list = ["├"]
    end_frame: list = ["└"]

    end_list: list = [frame]
    cp: int = len(maxi)-1
    for s in maxi:
        frame.append("─"*s)
        cut.append("─"*s)
        end_frame.append("─"*s)
        if cp:
            frame.append("┬")
            cut.append("┼")
            end_frame.append("┴")
            cp -= 1;

    frame.append("┐")
    cut.append("┤")
    end_frame.append("┘")

    for i in range(len(full_list)):
        tab: list = ["│"]
        for j in range(len(full_list[0])):
            if centered:
                val = (maxi[j]-len(str(full_list[i][j])))//2
                text = " "*val + str(full_list[i][j]) + " "*val
                if len(text) < maxi[j]:
                    text += " "
            else:
                val = maxi[j]-len(str(full_list[i][j]))-1
                text = " " + str(full_list[i][j]) + " "*val

            if left and j == 0:
                tab += [text] + ["║"]
            else:
                tab += [text] + ["│"]

        end_list.append(tab)

    end_list.append(end_frame)

    if labels:
        end_list.insert(2, cut)

    table: str = ""
    for i in range(len(frame)):
        table += "{"+str(i)+"}"
    table += "\n"

    answer: str = ""

    for elem in end_list:
        answer += table.format(*elem)

    return answer


if __name__ == "__main__":
    print(make_table(rows=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], labels=["e0", "e1", "e2"], left=["Index", 1, 2, 3]))
