import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from itertools import permutations


EXCEL_FILE = "suitcase_problem_instances.xlsx"
WEIGHT_LIMIT = 18
SIZE_SUM_LIMIT = 158
BIG_M = 200


def get_orientations(l, w, h):
    """
    Returns all unique rotations of one item.
    """
    return list(set(permutations([l, w, h], 3)))


def load_instance_data(excel_file, sheet_name, instance_number=None):

    #Load data and standardize column names
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Change these if your file uses different names
    rename_map = {
        "itemid": "item",
        "iteml_cm": "length",
        "itemw_cm": "width",
        "itemh_cm2": "height",
        "itemweight_kg2": "weight",
        "itemvalue": "value",
    }
    df = df.rename(columns=rename_map)

    # Filter by instance number if provided
    if instance_number is not None and "instance" in df.columns:
        df = df[df["instance"] == instance_number].reset_index(drop=True)

    required = ["length", "width", "height", "weight", "value"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "item" not in df.columns:
        df["item"] = [f"item_{i}" for i in range(len(df))]

    return df[["item", "length", "width", "height", "weight", "value"]]


def build_and_solve_model(df, instance_name="Instance"):
    # Convert dataframe to list of dicts for easier access
    items = df.to_dict("records")
    n = len(items)

    model = gp.Model(instance_name)

    # -- Decision variables --

    # Suitcase dimensions
    L = model.addVar(vtype=GRB.INTEGER, lb=1, ub=SIZE_SUM_LIMIT, name="L")
    W = model.addVar(vtype=GRB.INTEGER, lb=1, ub=SIZE_SUM_LIMIT, name="W")
    H = model.addVar(vtype=GRB.INTEGER, lb=1, ub=SIZE_SUM_LIMIT, name="H")

    # Binary: take item i or not
    take = model.addVars(n, vtype=GRB.BINARY, name="take")

    # Position of lower-left-bottom corner
    x = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="x")
    y = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="y")
    z = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="z")

    # Actual used dimensions after rotation
    lx = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="lx")
    ly = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="ly")
    lz = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="lz")

    # Orientation variables
    orientation_data = {}
    orient = {}

    # Precompute orientations for each item for faster access in constraints
    for i in range(n):
        dims = (
            int(items[i]["length"]),
            int(items[i]["width"]),
            int(items[i]["height"]),
        )
        rotations = get_orientations(*dims)
        orientation_data[i] = rotations

        for r in range(len(rotations)):
            orient[i, r] = model.addVar(vtype=GRB.BINARY, name=f"orient_{i}_{r}")

    model.update()

    # Objective -> maximize total value of packed items
    model.setObjective(
        gp.quicksum(items[i]["value"] * take[i] for i in range(n)),
        GRB.MAXIMIZE
    )

    # --Constraints--

    # Weight limit
    model.addConstr(
        gp.quicksum(items[i]["weight"] * take[i] for i in range(n)) <= WEIGHT_LIMIT,
        name="weight_limit"
    )

    # Suitcase size limit
    model.addConstr(L + W + H <= SIZE_SUM_LIMIT, name="size_sum_limit")

    # Symmetry -> enforce that there are no rotation variants of the suitcase itself
    model.addConstr(L >= W, name="sym1")
    model.addConstr(W >= H, name="sym2")

    for i in range(n):
        rotations = orientation_data[i]

        # If item is taken, choose exactly one orientation
        model.addConstr(
            gp.quicksum(orient[i, r] for r in range(len(rotations))) == take[i],
            name=f"one_orientation_{i}"
        )

        # Link chosen orientation to actual dimensions
        model.addConstr(
            lx[i] == gp.quicksum(rotations[r][0] * orient[i, r] for r in range(len(rotations))),
            name=f"link_lx_{i}"
        )
        model.addConstr(
            ly[i] == gp.quicksum(rotations[r][1] * orient[i, r] for r in range(len(rotations))),
            name=f"link_ly_{i}"
        )
        model.addConstr(
            lz[i] == gp.quicksum(rotations[r][2] * orient[i, r] for r in range(len(rotations))),
            name=f"link_lz_{i}"
        )

        # If packed, item must lie inside suitcase
        model.addConstr(x[i] + lx[i] <= L + BIG_M * (1 - take[i]), name=f"fit_x_{i}")
        model.addConstr(y[i] + ly[i] <= W + BIG_M * (1 - take[i]), name=f"fit_y_{i}")
        model.addConstr(z[i] + lz[i] <= H + BIG_M * (1 - take[i]), name=f"fit_z_{i}")

    # --Items cannot overlap -> for each pair of items, at least one of the 6 relative positions must hold
    for i in range(n):
        for j in range(i + 1, n):
            left_ij = model.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}")
            left_ji = model.addVar(vtype=GRB.BINARY, name=f"left_{j}_{i}")
            front_ij = model.addVar(vtype=GRB.BINARY, name=f"front_{i}_{j}")
            front_ji = model.addVar(vtype=GRB.BINARY, name=f"front_{j}_{i}")
            below_ij = model.addVar(vtype=GRB.BINARY, name=f"below_{i}_{j}")
            below_ji = model.addVar(vtype=GRB.BINARY, name=f"below_{j}_{i}")

            # If both items are selected, at least one relative position must hold
            model.addConstr(
                left_ij + left_ji + front_ij + front_ji + below_ij + below_ji
                >= take[i] + take[j] - 1,
                name=f"nonoverlap_logic_{i}_{j}"
            )

            model.addConstr(
                x[i] + lx[i] <= x[j] + BIG_M * (1 - left_ij),
                name=f"sep_x1_{i}_{j}"
            )
            model.addConstr(
                x[j] + lx[j] <= x[i] + BIG_M * (1 - left_ji),
                name=f"sep_x2_{i}_{j}"
            )
            model.addConstr(
                y[i] + ly[i] <= y[j] + BIG_M * (1 - front_ij),
                name=f"sep_y1_{i}_{j}"
            )
            model.addConstr(
                y[j] + ly[j] <= y[i] + BIG_M * (1 - front_ji),
                name=f"sep_y2_{i}_{j}"
            )
            model.addConstr(
                z[i] + lz[i] <= z[j] + BIG_M * (1 - below_ij),
                name=f"sep_z1_{i}_{j}"
            )
            model.addConstr(
                z[j] + lz[j] <= z[i] + BIG_M * (1 - below_ji),
                name=f"sep_z2_{i}_{j}"
            )

    # --Solve the model--
    model.Params.TimeLimit = 60

    # Hier mss nog wat parameter instellingen voor performance toevoegen

    model.write(f"{instance_name}.lp")  # write full model with all constraints
    model.optimize()

    return model, items, take, x, y, z, lx, ly, lz


def print_solution(model, items, take, x, y, z, lx, ly, lz):
    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        print("No usable solution found.")
        return

    print("\n================ SOLUTION ================\n")
    print(f"Objective value: {model.ObjVal:.2f}")
    print(f"Suitcase dimensions: L={model.getVarByName('L').X:.0f}, "
          f"W={model.getVarByName('W').X:.0f}, "
          f"H={model.getVarByName('H').X:.0f}")

    total_weight = 0
    for i in range(len(items)):
        if take[i].X > 0.5:
            total_weight += items[i]["weight"]
            print(
                f"{items[i]['item']}: "
                f"pos=({x[i].X:.0f}, {y[i].X:.0f}, {z[i].X:.0f}), "
                f"size=({lx[i].X:.0f}, {ly[i].X:.0f}, {lz[i].X:.0f}), "
                f"weight={items[i]['weight']}, value={items[i]['value']}"
            )

    print(f"Total weight: {total_weight}")


def main():
    xls = pd.ExcelFile(EXCEL_FILE)
    sheet_name = xls.sheet_names[0]

    # Discover all instance numbers
    all_df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
    all_df.columns = [str(c).strip().lower() for c in all_df.columns]
    instances = sorted(all_df["instance"].unique()) if "instance" in all_df.columns else [None]

    # Check for each instance
    for inst in instances:
        df = load_instance_data(EXCEL_FILE, sheet_name, instance_number=inst)
        instance_name = f"Instance_{inst}" if inst is not None else sheet_name
        print(f"\n--- Solving {instance_name} ---")
        model, items, take, x, y, z, lx, ly, lz = build_and_solve_model(df, instance_name)
        print_solution(model, items, take, x, y, z, lx, ly, lz)
        break


if __name__ == "__main__":
    main()