import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW1: 
            # Answer :

            # updating all input arguments.....
            new_input_args = [loma_ir.Arg(inpArg.id,
                                          autodiff.type_to_diff_type(diff_structs, inpArg.t), # mapping the arg ids to respective diff types
                                          inpArg.i)  for inpArg in node.args]
            
            # updating the return type of the function....
            new_ret_args = autodiff.type_to_diff_type(diff_structs, node.ret_type)



            # updating each element of the body of the function....
            new_body = []
            for stmt in node.body:
                new_body.append(self.mutate_stmt(stmt))
            
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            new_body = irmutator.flatten(new_body)


            # returning the updated components of the function....
            return loma_ir.FunctionDef(diff_func_id, 
                                       new_input_args, 
                                       new_body, 
                                       node.is_simd, 
                                       new_ret_args, 
                                       lineno = node.lineno)

        def mutate_return(self, node):
            # HW1: TODO
        
            # Updating the return statement...
            new_ret_statement = self.mutate_expr(node.val)
            
            # Handling the edge case of Int...
            if node.val.t == loma_ir.Int():
                return loma_ir.Return(new_ret_statement[0], lineno = node.lineno)
            
            # for struct return the class as it is...
            elif isinstance(node.val.t, loma_ir.Struct):
                return node

            # for all other cases return the dfloat...
            else:
                if new_ret_statement[1] is None:
                    return loma_ir.Return(new_ret_statement[0])
                else:
                    return loma_ir.Return(loma_ir.Call('make__dfloat', 
                                        [new_ret_statement[0], new_ret_statement[1]]), 
                                        lineno = node.lineno)

        def mutate_declare(self, node):
            # HW1: TODO

            # For struct declarations keep them as it is...
            if (node.val is not None) & (isinstance(node.t, loma_ir.Struct)):
                dec_r = node.val


            # Non Int cases are handled by the dfloat case...
            elif (node.val is not None) & (node.t != loma_ir.Int()):
                v1,v2 = self.mutate_expr(node.val)
                if v2 is None:
                    val = loma_ir.StructAccess(v1, 'val')
                    dval = loma_ir.StructAccess(v1, 'dval')
                    dec_r = loma_ir.Call('make__dfloat', [val, dval])
                else:
                    dec_r = loma_ir.Call('make__dfloat', [v1,v2])

            
            # in int cases return the node as it is
            elif (node.val is not None) & (node.t == loma_ir.Int()):
                return node
            else:
                dec_r = None

            
            return loma_ir.Declare(node.target,
                                   autodiff.type_to_diff_type(diff_structs, node.t),
                                   dec_r,
                                   lineno = node.lineno)

        def mutate_assign(self, node):
            # HW1: TODO

            # identifying the rhs expression 
            right_exp = self.mutate_expr(node.val)

            # using only value part not the differential part....
            if (isinstance(right_exp[1], loma_ir.ConstInt)):
                exp = right_exp[0]
            
            # for struct return as it is...
            elif (isinstance(right_exp[1], loma_ir.StructAccess)):
                return node
            else:
                exp  = loma_ir.Call('make__dfloat', right_exp)

            # handling LHS of assign op
            if isinstance(node.target, loma_ir.StructAccess):
                exp_t = loma_ir.StructAccess(node.target.struct, node.target.member_id)
                return loma_ir.Assign(exp_t, 
                                      exp,
                                      lineno = node.lineno)
            else:
                exp_t = self.mutate_expr(node.target)
                return loma_ir.Assign(exp_t[0].struct,
                                      exp,
                                      lineno = node.lineno)
        
        def mutate_ifelse(self, node):
            # HW3: TODO
            updated_if_else = loma_ir.IfElse(cond = self.mutate_expr(node.cond),
                                             then_stmts = irmutator.flatten([self.mutate_stmt(i) for i in node.then_stmts]),
                                             else_stmts = irmutator.flatten([self.mutate_stmt(i) for i in node.else_stmts]))

            return updated_if_else

        def mutate_while(self, node):
            # HW3: TODO
            # mutating thr body :
            cond_var = node.cond.left.id
            updated_body = []
            for i in node.body:
                if isinstance(i, loma_ir.Assign):
                    if i.target.id == cond_var:
                        updated_body.append(i)
                    else:
                        updated_body.append(self.mutate_stmt(i))

                else:
                    updated_body.append(self.mutate_stmt(i))

            updated_while = loma_ir.While(cond = node.cond,
                                          max_iter = node.max_iter,
                                          body = irmutator.flatten(updated_body))
            

            return updated_while

        def mutate_const_float(self, node):
            # HW1: TODO
            # direct differential return...
            return (node, loma_ir.ConstFloat(0.0))

        def mutate_const_int(self, node):
            # HW1: TODO
            # direct differential return...
            return (node, loma_ir.ConstInt(0))
        
        # overloading the greater than....
        def mutate_greater(self, node):
            l = self.mutate_expr(node.left)
            r = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(\
                loma_ir.Greater(),
                l[0],
                r[0],
                lineno = node.lineno,
                t = node.t)

        def mutate_var(self, node):
            # HW1: TODO
            # handling Int variables...
            if isinstance(node.t,loma_ir.Int):
                return (node, loma_ir.ConstInt(0))
            
            # handling the array case....
            elif isinstance(node.t, loma_ir.Array):
                return (node, None)

            # returning the fwd diff version
            return (loma_ir.StructAccess(node, 'val'), loma_ir.StructAccess(node, 'dval'))

        def mutate_array_access(self, node):
            # HW1: TODO

            # array & array index are handled....
            expr_1 = self.mutate_expr(node.array)
            expr_2 = self.mutate_expr(node.index)
            
            # int case direct return....
            if isinstance(node.array.t.t, loma_ir.Int):
                return (node, loma_ir.StructAccess(loma_ir.ArrayAccess(expr_1[0],
                                                                       expr_2[0],
                                                                       lineno = node.lineno,
                                                                       t = node.t), 
                                                    'dval'))

            val = loma_ir.StructAccess(loma_ir.ArrayAccess(expr_1[0],
                                                           expr_2[0],
                                                           lineno = node.lineno,
                                                           t = node.t), 
                                        'val')

            dval = loma_ir.StructAccess(loma_ir.ArrayAccess(expr_1[0],
                                       expr_2[0],
                                       lineno = node.lineno,
                                        t = node.t), 'dval')
            return (val,dval)

        def mutate_struct_access(self, node):
            # HW1: TODO

            # mutating struct members....
            memberVal = node.member_id
            entry = loma_ir.Var(node.member_id, t = node.t)
            mut_exp = self.mutate_expr(entry)

            # handling Int case
            if isinstance(mut_exp[1], loma_ir.ConstInt):
                return (node, mut_exp[1])
            
            # handling array case....
            elif (mut_exp[1] == None):
                return (loma_ir.StructAccess(node.struct, entry.id), None)
            
            val = loma_ir.StructAccess(loma_ir.StructAccess(node.struct, entry.id),'val')
            dval = loma_ir.StructAccess(loma_ir.StructAccess(node.struct, entry.id),'dval')
            return (val, dval)
                
        def mutate_add(self, node):
            # HW1: TODO
            left  = self.mutate_expr(node.left)
            right = self.mutate_expr(node.right)

            l_val = left[0]
            r_val = right[0]

            l_dval = left[1]
            r_dval = right[1]

            t_l_r_val = loma_ir.BinaryOp(loma_ir.Add(),
                                         l_val, r_val)

            d_t_l_r_val = loma_ir.BinaryOp(loma_ir.Add(),
                                           l_dval, r_dval)
            
            return (t_l_r_val, d_t_l_r_val)

        def mutate_sub(self, node):
            # HW1: TODO
            left  = self.mutate_expr(node.left)
            right = self.mutate_expr(node.right)

            l_val = left[0]
            r_val = right[0]

            l_dval = left[1]
            r_dval = right[1]

            t_l_r_val = loma_ir.BinaryOp(loma_ir.Sub(),
                                         l_val, r_val)

            d_t_l_r_val = loma_ir.BinaryOp(loma_ir.Sub(),
                                           l_dval, r_dval)
            
            
            return (t_l_r_val, d_t_l_r_val)

        def mutate_mul(self, node):
            # HW1: TODO
            left  = self.mutate_expr(node.left)

            right = self.mutate_expr(node.right)

            l_val = left[0]
            r_val = right[0]

            l_dval = left[1]
            r_dval = right[1]

            t_l_r_val = loma_ir.BinaryOp(loma_ir.Mul(),
                                         l_val, r_val)

            d_t_l_r_val = loma_ir.BinaryOp(loma_ir.Add(),
                                           loma_ir.BinaryOp(loma_ir.Mul(), l_dval, r_val),
                                           loma_ir.BinaryOp(loma_ir.Mul(), l_val, r_dval))
                
            
            return (t_l_r_val, d_t_l_r_val)

        def mutate_div(self, node):
            # HW1: TODO
            left  = self.mutate_expr(node.left)
            right = self.mutate_expr(node.right)

            l_val = left[0]
            r_val = right[0]

            l_dval = left[1]
            r_dval = right[1]

            t_l_r_val = loma_ir.BinaryOp(loma_ir.Div(),
                                         l_val, r_val)

            d_t_l_r_val = loma_ir.BinaryOp(loma_ir.Div(),
                                           loma_ir.BinaryOp(loma_ir.Sub(),
                                                            loma_ir.BinaryOp(loma_ir.Mul(), l_dval, r_val),
                                                            loma_ir.BinaryOp(loma_ir.Mul(), l_val, r_dval)),
                                           
                                           loma_ir.BinaryOp(loma_ir.Mul(), r_val, r_val))
                
            
            return (t_l_r_val, d_t_l_r_val)

        def mutate_call(self, node):
            # HW1: TODO
            # mapping the differentials....
            inpVar = []
            flag_arr = []
           

            for i in node.args:
                if isinstance(i, loma_ir.BinaryOp):
                    val, dval = self.mutate_expr(i)
                    inpVar.append((val, dval))
                    flag_arr.append(True)
                else:
                    inpVar.append(self.mutate_expr(i))
                    flag_arr.append(False)
  
    
            match node:
                case loma_ir.Call('sin'):
                    diff_value = loma_ir.BinaryOp(loma_ir.Mul(),
                                                  loma_ir.Call('cos', [inpVar[0][0]]),
                                                  inpVar[0][1])   

                    return  (loma_ir.Call('sin', [inpVar[0][0]]), diff_value)

                case loma_ir.Call('cos'):
                    diff_value = loma_ir.BinaryOp(loma_ir.Mul(),
                                                  loma_ir.BinaryOp(loma_ir.Mul(), 
                                                                   loma_ir.ConstInt(-1), 
                                                                   loma_ir.Call('sin', [inpVar[0][0]])),
                                                  inpVar[0][1])  

                    return  (loma_ir.Call('cos', [inpVar[0][0]]), diff_value)

                case loma_ir.Call('sqrt'):
                    diff_value = loma_ir.BinaryOp(loma_ir.Div(),
                                                  inpVar[0][1],
                                                  loma_ir.BinaryOp(loma_ir.Mul(), 
                                                                   loma_ir.ConstInt(2), 
                                                                   loma_ir.Call('sqrt', [inpVar[0][0]])))  

                    return  (loma_ir.Call('sqrt', [inpVar[0][0]]), diff_value)
                
                case loma_ir.Call('pow'):

                    tmp_res = loma_ir.BinaryOp(loma_ir.Mul(),    
                                                inpVar[1][0],
                                                loma_ir.Call('pow', [inpVar[0][0], 
                                                                     loma_ir.BinaryOp(loma_ir.Sub(), 
                                                                                      inpVar[1][0], 
                                                                                      loma_ir.ConstInt(1))]))
                    first_diff = loma_ir.BinaryOp(loma_ir.Mul(),    
                                                  inpVar[0][1],
                                                  tmp_res)


                    log_expr = loma_ir.Call('log', [inpVar[0][0]])
                    pow_expr = loma_ir.Call('pow', [inpVar[0][0], inpVar[1][0]])

                    tmp_prod = loma_ir.BinaryOp(loma_ir.Mul(),
                                                pow_expr,
                                                log_expr)
                            
                    second_diff = loma_ir.BinaryOp(loma_ir.Mul(),
                                                   inpVar[1][1],
                                                   tmp_prod)
                                


                    diff_value = loma_ir.BinaryOp(loma_ir.Add(),
                                                  first_diff,
                                                  second_diff)  

                    return  (loma_ir.Call('pow', [inpVar[0][0], inpVar[1][0]]), diff_value)

                case loma_ir.Call('exp'):
                    diff_val = loma_ir.BinaryOp(loma_ir.Mul(),    
                                                loma_ir.Call('exp',[inpVar[0][0]]),
                                                inpVar[0][1])
                    
                    return  (loma_ir.Call('exp',[inpVar[0][0]]), diff_val)
                
                case loma_ir.Call('log'):
                    diff_val = loma_ir.BinaryOp(loma_ir.Div(),   
                                                inpVar[0][1],
                                                inpVar[0][0])
                    
                    return (loma_ir.Call('log',[inpVar[0][0]]), diff_val)

                case loma_ir.Call('float2int'):
                    return (loma_ir.Call('float2int', [inpVar[0][0]], t = node.t ), loma_ir.ConstInt(0))
                
                case loma_ir.Call('int2float'):
                    return (loma_ir.Call('int2float', [inpVar[0][0]], t = node.t ), loma_ir.ConstFloat(0.0))
                
                case _:
                    updated_args = []
                    arg_props = [i for i in funcs[node.id].args]
                    for i in range(len(inpVar)):
                        if isinstance(arg_props[i].i, loma_ir.Out):
                            updated_args.append(node.args[i])
                            continue
                        
                        if isinstance(inpVar[i][0], loma_ir.Call):
                            updated_args.append(inpVar[i][0])
                            continue
                        updated_args.append(loma_ir.Call('make__dfloat', [inpVar[i][0], inpVar[i][1]])) 
                    
            
                    new_func = loma_ir.Call(func_to_fwd[node.id], irmutator.flatten(updated_args))
                 
                    return new_func, None

        def mutate_call_stmt(self, node):
            new_func, _ = self.mutate_expr(node.call)
            return loma_ir.CallStmt(new_func, lineno = node.lineno)

       


    return FwdDiffMutator().mutate_function_def(func)











