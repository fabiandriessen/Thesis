

def update_feasible_combinations(feasible_combinations, df_abm):

    charging_points = list(df_abm.loc[(df_abm.charging_stations.notna()) &
                                      (df_abm.charging_stations != 0)].name.unique())
    updated_feasible_combinations = {}
    for key, combinations in feasible_combinations.items():
        updated_feasible_combinations[key] = []
        if len(combinations) > 1:
            for combi in combinations:
                # print(combi)
                feasible = True
                for node in combi:
                    if not node in charging_points:
                        feasible = False
                        break
                if feasible:
                    updated_feasible_combinations[key].append(combi)
        else:
            feasible = True
            for node in combinations:
                if node in charging_points:
                    pass
                else:
                    feasible = False
                    break
            if feasible:
                updated_feasible_combinations[key].append(combinations)

    return updated_feasible_combinations
