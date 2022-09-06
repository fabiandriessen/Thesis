from pulp import *
import re
import pickle


def second_stage_frlm(r, v, b, p, c, max_per_loc, o, df_g, df_b, df_eq_fq):
    """ This program optimally sites p charging stations with a max capacity c,
    based on three DataFrames that are generated by the first_stage_FRLM function.
        Parameters
        ----------
        r : int
            vessel range

        v : int
            travel speed resulting in the range

        b : int
            power of basis vessel [M1]

        p : int
            charging stations modules to locate on any node of G.

        c : float
            max charging capacity of a charging station

        max_per_loc: int
        Maximum number of charging modules that can be placed at a certain location.

        o: float
            operational hours of a charging station during same time period as c

        ## the following three inputs are data frames that can be generated using the first_stage_FRLM function
        df_g : pandas.DataFrame
            A dataFrame with a row for each route q and a column for each existing charging station combination h.
            b_qh = 1 combination h can support trips on path h, 0 otherwise.

        df_b : pandas.DataFrame
            A dataFrame with a row for each charging station combination h, and a column for each unique facility k.
            a_qh = 1 if combination k is in combination h, 0 otherwise.

        df_eq_fq : pandas.DataFrame
            A dataframe with a row for each route q, that contains two columns corresponding f_q and e_q values.
        """

    # define y_qh for each q and each h, and restrict between 0 and 1
    # constraint 1.5 already incorporated
    # create list of index to be able to loop over double index

    # daily capacity equal to the number of small vessels a station can serve
    daily_cap = (o*c*v)/(r*b)

    a = df_g.reset_index()
    flow_a = []
    for i in a.index:
        flow_a.append((a.q[i], a.h[i]))

    # first decision variable: allocated flow
    flow_allocation = pulp.LpVariable.dicts("Flow_captured",
                                            ((q, h) for q, h in flow_a),
                                            lowBound=0,
                                            upBound=1,
                                            cat='Continuous')

    # second decision variable: number of facilities to place at each site is also decision var
    facilities_to_build = pulp.LpVariable.dicts("Facilities",
                                                (facility for facility in df_g.columns),
                                                lowBound=0,
                                                upBound=max_per_loc,
                                                cat='Integer')
    # problem definition
    model = LpProblem('CFRLM', LpMaximize)

    # objective function
    model += pulp.lpSum([flow_allocation[q, h] * df_b[h][q] * df_eq_fq['f_q'][q] for q, h in flow_a])

    # ###############################################constraints##################################################
    # first constraint
    # for each facility
    for key, facility in facilities_to_build.items():
        model += pulp.lpSum(df_eq_fq['e_q'][q] * df_g[key].loc[df_g.index == (q, h)] * df_eq_fq['f_q'][q] *
                            flow_allocation[q, h] for q, h in df_g.index) <= pulp.lpSum(daily_cap * facility)

    # second constraint
    model += pulp.lpSum(facilities_to_build[i] for i in facilities_to_build.keys()) <= p

    # third constraint
    for q in df_b.index:
        model += pulp.lpSum([flow_allocation[q, h] * df_b[h][q]] for h in df_g.reset_index().
                            loc[df_g.reset_index().q == q].h) <= 1

    # print(model)

    # solve
    model.solve()

    status = LpStatus[model.status]
    print(status)
    # Values of decision variables at optimum

    # for var in model.variables():
    #     print('Variable', var, 'is equal to', value(var))
    #
    # # Value of objective at optimum
    # print('Total supported flow is equal to', value(model.objective))

    # create useful outputs
    output_dict = {}
    # fill this dict
    for var in model.variables():
        output_dict[str(var)] = value(var)
    # tuple(re.sub('''["'_']''', "", i[16:35]).split(','))
    # now divide over dicts
    optimal_facilities = {re.sub('[^0-9]', "", i): output_dict[i] for i in output_dict.keys() if 'Facilities' in i}
    optimal_flows = {i: output_dict[i] for i in output_dict.keys() if 'captured' in i}

    non_zero_flows = {}
    for key, item in optimal_flows.items():
        if item != 0:
            a = (re.sub('''["'_']''', "", key[16:35]).split(','))
            a = tuple([a[0], a[1], int(a[2])])
            non_zero_flows[a] = {'combinations': [], 'flows': []}

    for key, item in optimal_flows.items():
        if item != 0:
            a = (re.sub('''["'_']''', "", key[16:35]).split(','))
            a = tuple([a[0], a[1], int(a[2])])
            b = key[40:-3].split("',_'")
            if len(b) == 1:
                b[0] = re.sub("[']", "", b[0])
            non_zero_flows[a]['combinations'].append(b)
            non_zero_flows[a]['flows'].append(item)

    routes_supported = float(len(non_zero_flows.keys()))

    return optimal_facilities, optimal_flows, non_zero_flows, value(model.objective), routes_supported
