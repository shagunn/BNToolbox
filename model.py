import sys
import pprint
import itertools
import copy
import pandas as pd
import decimal
# import numpy as np

class Model:
    # input nets in the following form:
    # 	[(p1, c1), (p2, c2), ..., (pn, cn)]
    #	where p is a parent node and c is a child node
    def __init__(self, nets):
        self.nodes = {}
        # for all nodes in nets, parse unique nodes
        for p, c in nets:
            if p not in self.nodes:
                self.nodes[p] = {'parents':[], 'children':[]}
            if c not in self.nodes:
                self.nodes[c] = {'parents':[], 'children':[]}

            self.nodes[p]['children'].append(c)
            self.nodes[c]['parents'].append(p)

    def prettyPrint(self, d):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(d)
        # for k, v in self.nodes.items():
            # print(k, v)

    def addCPD(self, cpd):
        n = cpd['node']
        if n not in self.nodes:
            print("Node {} not defined.".format(n))
            sys.exit(1)
        self.nodes[n].update({
            'cpt'  : cpd['cpt'],
            'evid' : cpd['evid'],
        })



    # calculate edges for each node
    # not being used currently
    # def _add_edges(self):
    #     for node in self.nodes:
    #         count_children = len(self.nodes[node]['children'])
    #         count_parents = len(self.nodes[node]['parents'])
    #         edges = count_children + count_parents
    #
    #         self.nodes[node].update({'edges': edges})

    def _make_dataframe(self, answer):

        string = ''.join(answer[0]['vars'])

        columns = []
        columns = answer[0]['vars']
        columns.append('Pr')

        ans_0 = decimal.Decimal(answer[0]['Pr'][0])
        ans_1 = decimal.Decimal(answer[0]['Pr'][1])

        df = pd.DataFrame(index=[0, 1], columns=columns)
        df.iat[0,0] = string + '_0'  #index (row), column
        df.iat[1,0] = string + '_1'
        df.iat[0,1] = round(ans_0, 4)
        df.iat[1, 1] = round(ans_1, 4)
        return df
        # prob_list = []
        # label = []
        # df_cpt = {}
        #
        # for node in self.nodes:
        #     # get column labels
        #     label.append(node)
        #     if self.nodes[node]['evid'] != None:
        #         label.extend(self.nodes[node]['evid'])
        #
        #     # get 01 pattern
        #     order = self._getpattern(node)
        #
        #     # create dataframe
        #     df_cpt[node] = pd.DataFrame.from_records(order, columns=label)
        #
        #     # convert probabilities from cpt_dict to list
        #     for prob in self.nodes[node]['cpt']:
        #         prob_list.extend(prob)
        #     prob_se = pd.Series(prob_list)
        #
        #     # append probabilities to df
        #     df_cpt[node]['Pr'] = prob_se.values
        #
        #     # reset lists
        #     label = []
        #     prob_list = []
        #
        # # test print dataframes
        # # for node in self.nodes:
        # #     print(df_cpt[node])
        #
        # return df_cpt


    # Query Format: myModel.query('J', e={'G': 0, 'R': 1})
    def query(self, Y, e=None):
        # P(Y)
        # P(Y| e = e1)
        if not Y:
            print("No query variable specified")
            sys.exit(1)
        if Y not in self.nodes:
            print("Unknown variable: {}".format(Y))
            sys.exit(1)

        for key in e:
            if key not in self.nodes:
                print("Unknown evidence: {}".format(key))
                sys.exit(1)

        self._make_factors()
        # self.prettyPrint(self.factors)

        if e != None:
            self.factors = self._update_factor_e(e, self.factors)

        # print("UPDATED FACTORS (if evidence given): ")
        # self.prettyPrint(self.factors)

        ans_cpt = {}
        ans_cpt = _var_elim(Y, e, self.factors, self._var_elim_order())

        # NORMALIZE
        if e != None:
            multiplication_steps = []
            if len(ans_cpt) > 1:
                if len(ans_cpt) == 2:
                    ans_cpt.append(self._norm_multiply(ans_cpt[0], ans_cpt[1]))  ##### CHANGE THIS
                    del ans_cpt[0]
                    del ans_cpt[0]
                    # print('ans_cpt', ans_cpt)
                else:
                    multiplication_steps = len(ans_cpt) - 1
                    for step in range(multiplication_steps):
                        ans_cpt.append(self._norm_multiply(ans_cpt[0], ans_cpt[1]))  ##### CHANGE THIS
                        if len(ans_cpt) != 1:
                            del ans_cpt[0]
                            del ans_cpt[0]
            ans_cpt.append(self._normalize(ans_cpt[0]))
            del ans_cpt[0]

        return self._make_dataframe(ans_cpt)

    def _norm_multiply(self, f1, f2):
        final_Pr = []
        f1_Pr = f1['Pr']
        f2_Pr = f2['Pr']
        final_Pr = [a*b for a,b in zip(f1_Pr,f2_Pr)]
        multiplied_result = {}
        multiplied_result['vars'] = f1['vars']
        multiplied_result['Pr'] = final_Pr
        return multiplied_result

    def _normalize(self, cpt):
        Pr_list =  cpt['Pr']
        Pr_sum = sum(Pr_list)
        Pr_final = [x / Pr_sum for x in Pr_list]

        Pr_result = {}
        Pr_result['vars'] = cpt['vars']
        Pr_result['Pr'] = Pr_final

        return Pr_result





    def _make_factors(self):
        self.factors = []
        for node in self.nodes:
            # print(self.nodes[node])
            var = [node]
            if self.nodes[node]['evid']:
                var.extend(self.nodes[node]['evid'])
            cpt = list(itertools.chain.from_iterable(self.nodes[node]['cpt']))
            factor = {}
            factor['vars'] = var
            factor['Pr'] = cpt
            self.factors.append(factor)

    def _update_factor_e(self, e, factors):
        for key in e:       # iterate through evidence variable
            # print ('KEY:', key)     # VARIABLE
            # print(e[key])   # VALUE
            elim_factors = []
            factorcounter = 0
            factors_to_append = []
            key_list = []
            for factor in factors:  # iterate through factors
                # print('current factor:', factor)
                for var in factor['vars']: # iterate through each variable in given factor
                    if var == key:

                        # find which factors are being modified, and make a list of their positions so that they can be deleted
                        if factorcounter not in elim_factors:
                            elim_factors.append(factorcounter)

                        # if the factor value needs to be modified:
                        if len(factor['vars']) > 1:
                            modfactor_index = _getpattern(len(factor['vars']))
                            f_Pr = factor['Pr']
                            # print('modfactor_index', modfactor_index)
                            eliminate_mf = _f1f2_eliminate(factor['vars'], key)
                            # print('eliminate_mf', eliminate_mf)

                            evid_var_index = []
                            evid_var_index = copy.deepcopy(modfactor_index)
                            evid_var_index, evid_var_list = _f1f2_orderedcpt(eliminate_mf, evid_var_index, factor['vars'])
                            # print(evid_var_index)

                            modfactor_index_keep = []
                            for row in range(len(evid_var_index)):
                                if evid_var_index[row][0] == e[key]:
                                    modfactor_index_keep.append(row)

                            # print('modfactor_index_keep', modfactor_index_keep)
                            f_new_Pr = []
                            f_new_Pr = [prob for idx, prob in enumerate(f_Pr) if idx in modfactor_index_keep]
                            new_cpt = []
                            new_cpt = [row for idx, row in enumerate(modfactor_index) if idx in modfactor_index_keep]
                            key_list = []
                            key_list.append(key)
                            new_cpt, new_var_list = _f1f2_orderedcpt(key_list, new_cpt, factor['vars'])

                            final_index = _getpattern(len(new_var_list))
                            final_pr = []

                            for row in final_index:
                                if row in new_cpt:
                                    final_pr.append(f_new_Pr[new_cpt.index(row)])



                            final_factor = {}
                            final_factor['vars'] = new_var_list
                            final_factor['Pr'] = final_pr


                            factors_to_append.append(final_factor)
                factorcounter = factorcounter + 1
            factors = [row for idx, row in enumerate(factors) if idx not in elim_factors]



            factors.extend(factors_to_append)
            # print('FINAAAAAAAAL FACTORS:', factors)

        return factors




    def _var_elim_order(self):
        # min-neighbours
        # eliminate any variables with no parents first
        # then eliminate variables with no children
        # then eliminate anything with least number of edges?
        # NEED TO FIX!!

        elim_order = []
        var_list = []

        var_list = list(self.nodes.keys())

        #count parents
        for node in self.nodes:
            if len(self.nodes[node]['children']) == 0:
                if len(self.nodes[node]['parents']) == 1:
                    elim_order.append(node)
                    var_list.remove(node)

        for node in self.nodes:
            if len(self.nodes[node]['parents']) == 0:
                elim_order.append(node)
                var_list.remove(node)

        remaining = var_list

        for node in remaining:
            if len(self.nodes[node]['children']) == 0:
                elim_order.append(node)
                var_list.remove(node)

        elim_order.extend(var_list)

        return elim_order



# ----- Non Class Private Methods ----- #
def _var_elim(Y, e, factors, elim_order=None):

    elim_list = elim_order
    elim_list.remove(Y) # remove var being queried
    for key in e:
        elim_list.remove(key)
    # print('ELIMINATION ORDER:', elim_list)

    for var_elim in elim_list:
        # print("       " )
        # print("VARIABLE BEING ELIMINATED:",var_elim )
        fct_elim = []
        fct_not_elim = []
        for factor in factors:
            append_counter = 0
            for var in factor['vars']:
                if var == var_elim:
                    fct_elim.append(factor)
                    append_counter =+1
            if append_counter == 0:
                fct_not_elim.append(factor)
        # print("FACTORS NOT BEING ELIMINATED:")
        # print(fct_not_elim)
        factors = []
        factors = fct_not_elim
        multiply_return = []
        multiply_return = _multiply(var_elim, fct_elim)
        # print("RETURN VALUE FROM MULTIPLY FUNCTION (SUMMED OUT PR):")
        # print(multiply_return)

        if multiply_return != None:
            factors.append(multiply_return)

        # print("FACTORS NOT ELIM + MULTIPLY RETURN:")
        # print(factors)

    return factors




def _multiply(var, fct_list):
    # print('CONFIRM var being eliminated for _multiply:', var)
    # print('FACTORS given for multiplication:', fct_list)
    checklist = []
    for l in range(len(fct_list)):
        for letter in fct_list[l]['vars']:
            if letter not in checklist:
                checklist.extend(letter)

    if len(checklist) > 1:
        if len(fct_list) >= 2:
            if len(fct_list) == 2:
                # print("FOR 2 FACTORS BEING MULTIPLIED")
                # print(fct_list)
                m_factor, p_cpt = _multiply_two(var, fct_list[0], fct_list[1])
                return _sum_out(m_factor, p_cpt, var)
            else:
                multiplication_steps = len(fct_list) - 1
                # print("FOR MORE THAN 2 FACTORS BEING MULTIPLIED")
                # print(fct_list)
                for step in range(multiplication_steps):
                    m_factor, p_cpt = _multiply_two(var, fct_list[0], fct_list[1])
                    fct_list.append(m_factor)
                    if len(fct_list) != 1:
                        del fct_list[0]
                        del fct_list[0]
                return _sum_out(fct_list[0], p_cpt, var)
                # return fct_list[0]


def _multiply_two(var, fct1, fct2):

    # print('CONFIRM var being eliminated for _multiply_two:', var)
    f1 = fct1['vars']
    f2 = fct2['vars']

    f1_Pr = fct1['Pr']
    f2_Pr = fct2['Pr']
    # print('f1', f1)
    # print('f2', f2)
    # print('f1_Pr', f1_Pr)
    # print('f2_Pr', f2_Pr)

    f1_index = _getpattern(len(f1))
    f2_index = _getpattern(len(f2))
    # print('PAT1', f1_index)
    # print('PAT2', f2_index)

    combinedlist = []
    combinedlist.extend(f1)
    combinedlist.extend(f2val for f2val in f2 if f2val not in f1)
    # print('combinedlist', combinedlist)

    prod_cpt = []
    prod_cpt = _getpattern(len(combinedlist))
    # print('prod_cpt PAT')
    # print(prod_cpt)

    eliminate_f1 = []
    eliminate_f1 = _f1f2_eliminate(combinedlist, f1)
    # print('print(eliminate_f1):', eliminate_f1)

    prod_cpt_f1 = []
    prod_cpt_f1 = copy.deepcopy(prod_cpt)
    prod_cpt_f1, f1_prod_list = _f1f2_orderedcpt(eliminate_f1, prod_cpt_f1, combinedlist)

    prod_f1_Pr = []
    for row in prod_cpt_f1:
        if row in f1_index:
            prod_f1_Pr.append(f1_Pr[f1_index.index(row)])

    ### ----- F2 -----
    eliminate_f2 = []
    eliminate_f2 = _f1f2_eliminate(combinedlist, f2)
    # print('print(eliminate_f2):',eliminate_f2)

    prod_cpt_f2 = []
    prod_cpt_f2 = copy.deepcopy(prod_cpt)
    prod_cpt_f2, f2_prod_list = _f1f2_orderedcpt(eliminate_f2, prod_cpt_f2, combinedlist)
    # print('f2_prod_list', f2_prod_list)
    # print("prod_cpt_f2", prod_cpt_f2)
    prod_cpt_f2_reordered = []

    prod_cpt_f2_reordered = _reorder_columns(prod_cpt_f2, _getorder(f2, f2_prod_list))

    # print('prod_cpt_f2_reordered', prod_cpt_f2_reordered)

    prod_f2_Pr = []
    for row in prod_cpt_f2_reordered:
        if row in f2_index:
            prod_f2_Pr.append(f2_Pr[f2_index.index(row)])
    # print('prod_f2_Pr', prod_f2_Pr)

    ## ----- MULTIPLY ----
    new_Pr = []
    new_Pr = [a*b for a,b in zip(prod_f1_Pr,prod_f2_Pr)]


    ## --- RETURN FORMAT ----
    multiplied_factor = {}
    multiplied_factor['vars'] = combinedlist
    multiplied_factor['Pr'] = new_Pr

    # print(new_factor)

    return multiplied_factor, prod_cpt


def _sum_out(multiplied_factor, prod_cpt, var):
    combinedlist = []
    combinedlist = multiplied_factor['vars']
    multiplied_Pr = multiplied_factor['Pr']
    sumout_index_length = len(combinedlist) - 1
    sumout_index = _getpattern(sumout_index_length)


    cpt_sumout = []
    cpt_sumout = copy.deepcopy(prod_cpt)

    sumoutelim = []
    sumoutelim.append(var)

    cpt_sumout, sumout_list = _f1f2_orderedcpt(sumoutelim, cpt_sumout, combinedlist)
    summedout_Pr = [[] for i in range(int(len(prod_cpt)/2))]

    rowcounter = 0
    for row in cpt_sumout:
        if row in sumout_index:
            summedout_Pr[sumout_index.index(row)].append(multiplied_Pr[rowcounter])
        rowcounter += 1

    sumresult = [sum(b) for b in summedout_Pr]


    new_factor = {}
    new_factor['vars'] = sumout_list
    new_factor['Pr'] = sumresult

    # print(new_factor)

    return new_factor


def _get_index(var, combinedlist_f):
    return combinedlist_f.index(var)

def _f1f2_eliminate(cl, factor):
    eliminate = []
    for variable in cl:
        if variable not in factor:
            eliminate.append(variable)
    return eliminate

def _f1f2_orderedcpt(eliminate, f_prod_cpt, allvars):
    f_combinedlist = []
    f_combinedlist.extend(allvars)

    for variable in eliminate:
        id_value = _get_index(variable, f_combinedlist)
        for row in f_prod_cpt:
            del row[id_value]
        del f_combinedlist[id_value]
    return f_prod_cpt, f_combinedlist

# change order
def _getorder(curr_order, orig_order):
    myorder = []
    for var in curr_order:
        myorder.append(_get_index(var, orig_order))
    return myorder

def _reorder_columns(listofRows, column_indexes):
    # line of code below from python cookbook - https://www.safaribooksonline.com/library/view/python-cookbook-2nd/0596007973/ch04s08.html
    return [ [row[ci] for ci in column_indexes] for row in listofRows ]

# ORGINAL CODE:
# def _getpattern(n):
#     # d = (len(self.nodes[node]['parents'])) + 1
#     pattern = list(itertools.product([0,1], repeat=n))
#     return pattern
# NEW CODE (written with help from Paul B.):
def _getpattern(n):
  if n < 1:
    return [[]]
  sub_cpt = _getpattern(n-1)
  return [ row + [v] for row in sub_cpt for v in (0, 1)]
