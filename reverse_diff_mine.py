import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random


# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

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
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target, deriv, overwrite):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    return [loma_ir.Assign(target,
                        loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False
    
    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            self.seen_variables = []
            for i in node.args:
                if i.i == loma_ir.In():
                    self.seen_variables.append(i.id)
     
            self.call_var_to_func_mapping = {}
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)
            

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
           
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]
            

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            
            self.seen_variables.append(node.target)

            return self.tmp_assign_stmts + [loma_ir.Declare(\
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            val = self.mutate_expr(node.val)
            if isinstance(node.target, loma_ir.ArrayAccess):
                self.seen_variables.append(node.target.array.id)
            else:
                if not isinstance(node.target, loma_ir.StructAccess):
                    self.seen_variables.append(node.target.id)
           
            return self.tmp_assign_stmts + [loma_ir.Assign(\
                target,
                val,
                lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

        def mutate_call(self, node):

            if funcs.get(node.id, "") != "":
                og_func_args = funcs[node.id].args
            else:
                og_func_args = [loma_ir.Arg("tmp", t = loma_ir.Float(), i = loma_ir.In())]*len(node.args)

            new_args = []
            for c,arg in enumerate(node.args):
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    
             
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)

                elif (og_func_args[c].i == loma_ir.Out()):
                    if (node.args[c].id not in self.seen_variables) & (not isinstance(arg.t, loma_ir.Array)):

                        # tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                        # self.tmp_count += 1
                        # tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                        # self.tmp_declare_stmts.append(loma_ir.Declare(\
                        #     tmp_name, arg.t))
                        # self.tmp_assign_stmts.append(loma_ir.Assign(\
                        #     tmp_var, loma_ir.Var("_d" + arg.id)))

                        # self.call_var_to_func_mapping[tmp_var.id] = "_d" + arg.id
                        # new_args.append(tmp_var)
                        new_args.append(arg)
                    else:
                        # tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                        # self.tmp_count += 1
                        # tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                        # self.tmp_declare_stmts.append(loma_ir.Declare(\
                        #     tmp_name, arg.t))
                        # self.tmp_assign_stmts.append(loma_ir.Assign(\
                        #     tmp_var, loma_ir.Var(arg.id)))
                        
                        # new_args.append(tmp_var)
                        new_args.append(arg)


                else:
                    new_args.append(arg)
                


            return loma_ir.Call(node.id, new_args, t = node.t)

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.

    # Forward pass mutator
    class FwdPassMutator(irmutator.IRMutator):
        def __init__(self, count, out_args, loop_vars, seen_vars):
            self.s_size = count
            self.input_var_cnt = 0
            self.fwd_pass_tape = []
            self.data_types_list = []
            self.output_args = out_args
            self.loop_ctr_cnt = 0
            self.loop_vars = loop_vars
            self.level_while = 0
            self.elements_inside_loop = []
            self.seen_vars = seen_vars


            self.tape = loma_ir.Declare(target = "tape", 
                                         t = loma_ir.Array(t = loma_ir.Float(), 
                                                           static_size = count))
            self.tape_ctr = loma_ir.Declare(target = 'tape_ctr', 
                                             t = loma_ir.Int(), 
                                             val = loma_ir.ConstInt(0))

        def check_lhs_is_output_arg(self, lhs):
            match lhs:
                case loma_ir.Var():
                    return lhs.id in self.output_args
                case loma_ir.StructAccess():
                    return self.check_lhs_is_output_arg(lhs.struct)
                case loma_ir.ArrayAccess():
                    return self.check_lhs_is_output_arg(lhs.array)
                case _:
                    assert False
            
        def mutate_function_def(self, node):
            self.dec_stmts = [self.tape, self.tape_ctr]

            for l_c, s in enumerate(node.body):
                # does nothing because we inheriting from the og code...

                if isinstance(s, loma_ir.Declare):
                    self.dec_stmts.append(s)
                    self.mutate_declare(s)
                    self.data_types_list.append(s.t)


                    # accessing array elements to store value
                    if (l_c+1) != (len(node.body)-1):
                        if isinstance(s.t, loma_ir.Struct):
                            continue
                        
                        
                        assig_st = loma_ir.Assign(target = loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                        loma_ir.Var("tape_ctr")),
                                                val = loma_ir.Var(s.target))
                        self.dec_stmts.append(assig_st)
                        self.fwd_pass_tape.append(s.target)

                        # increasing the counter....
                        updated_st = loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                                    val = loma_ir.BinaryOp(loma_ir.Add(),
                                                                        loma_ir.Var("tape_ctr"),
                                                                        loma_ir.ConstInt(1)))
                        self.dec_stmts.append(updated_st)

                elif isinstance(s, loma_ir.Return) is False:

                    if isinstance(s, loma_ir.IfElse) is True:
                        self.dec_stmts.append(self.mutate_ifelse(s))
                        continue

                    if isinstance(s, loma_ir.While) is True:
                        
                        self.dec_stmts.append(loma_ir.Declare(target = 'loop_ctr' + str(0),
                                              t = loma_ir.Int(),
                                              val = loma_ir.ConstInt(0)))

                        self.dec_stmts.append(loma_ir.Declare(target ='loop_ctr' + str(1), 
                                         t = loma_ir.Array(t = loma_ir.Float(), 
                                                           static_size = self.s_size)))

                        self.dec_stmts.append(loma_ir.Declare(target = 'loop_ptr' + str(1),
                                              t = loma_ir.Int(),
                                              val = loma_ir.ConstInt(0)))
                        
                        self.dec_stmts.append(loma_ir.Declare(target = 'loop_tmp' + str(1),
                                              t = loma_ir.Int(),
                                              val = loma_ir.ConstInt(0)))

                        
                        self.dec_stmts.append(loma_ir.Declare(target ='loop_ctr' + str(2), 
                                         t = loma_ir.Array(t = loma_ir.Float(), 
                                                           static_size = self.s_size)))
                        
                        self.dec_stmts.append(loma_ir.Declare(target = 'loop_ptr' + str(2),
                                              t = loma_ir.Int(),
                                              val = loma_ir.ConstInt(0)))
                        
                        self.dec_stmts.append(loma_ir.Declare(target = 'loop_tmp' + str(2),
                                              t = loma_ir.Int(),
                                              val = loma_ir.ConstInt(0)))

                        

                        self.dec_stmts.append(self.mutate_while(s))
                        continue
                    
                    if isinstance(s, loma_ir.CallStmt):
                        # argument list....
                        flg = False
                        for a_i in s.call.args:
                            if a_i.id not in self.seen_vars:
                              flg = True
                              break
                              
                        if flg:
                            continue

                        prev_c = self.input_var_cnt
                        op = self.mutate_call_stmt(s)
                        new_c = self.input_var_cnt
                        self.dec_stmts.append(s)
                        self.data_types_list.extend([loma_ir.Float()]*(new_c - prev_c))
                        continue

                    

                    if self.check_lhs_is_output_arg(s.target):
                        prev_c = self.input_var_cnt
                        s = self.mutate_stmt(s)
                        new_c = self.input_var_cnt
                        self.data_types_list.extend([s.target.t]*(new_c - prev_c))
                        continue
                
                    prev_c = self.input_var_cnt
                    self.dec_stmts.append(self.mutate_stmt(s))
                    new_c = self.input_var_cnt
      

                    self.data_types_list.extend([s.target.t]*(new_c - prev_c))
                    
                    # accessing array elements to store value
                    if (l_c+1) != (len(node.body)-1):
     
                        if isinstance(s.target.t, loma_ir.Struct):
                            continue

                        if isinstance(s.target, loma_ir.StructAccess):
                            assig_st = loma_ir.Assign(target = loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                            loma_ir.Var("tape_ctr")),
                                                    val = s.target)
                            self.dec_stmts.append(assig_st)
                            self.fwd_pass_tape.append(s.target)
                        
                        else:
                            assig_st = loma_ir.Assign(target = loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                            loma_ir.Var("tape_ctr")),
                                                    val = loma_ir.Var(s.target.id))
                            self.dec_stmts.append(assig_st)
                            self.fwd_pass_tape.append(s.target.id)

                        # increasing the counter....
                        updated_st = loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                                    val = loma_ir.BinaryOp(loma_ir.Add(),
                                                                        loma_ir.Var("tape_ctr"),
                                                                        loma_ir.ConstInt(1)))
                        self.dec_stmts.append(updated_st)
        
        def mutate_declare(self, node):
            s = loma_ir.Declare("_d" + node.target, node.t)
            self.dec_stmts.append(s)
      
        # overloaded function to store count of inputs for backprop....
        def mutate_var(self, node):
            self.input_var_cnt+=1
            return node

        # overload if else case....
        def mutate_ifelse(self, node):
            new_cond = node.cond
            new_then_stmts = []
            for i in node.then_stmts: 
                if isinstance(i, loma_ir.IfElse):
                    r = self.mutate_ifelse(i)
                    new_then_stmts.append(r)
                    continue
                
                self.data_types_list.extend([i.target.t])
                self.input_var_cnt+=1
                assig_st = loma_ir.Assign(target = loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                       loma_ir.Var("tape_ctr")),
                                                                       val = loma_ir.Var(i.target.id))
                new_then_stmts.append(assig_st)
                self.fwd_pass_tape.append(i.target.id)

                # increasing the counter....
                updated_st = loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                            val = loma_ir.BinaryOp(loma_ir.Add(),
                                                                    loma_ir.Var("tape_ctr"),
                                                                    loma_ir.ConstInt(1)))
                new_then_stmts.append(updated_st)
                new_then_stmts.append(i)
            
            new_else_stmts = []
            for i in node.else_stmts: 
                if isinstance(i, loma_ir.IfElse):
                    r = self.mutate_ifelse(i)
                    new_else_stmts.append(r)
                    continue

                self.data_types_list.extend([i.target.t])
                self.input_var_cnt+=1
                assig_st = loma_ir.Assign(target = loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                       loma_ir.Var("tape_ctr")),
                                                                       val = loma_ir.Var(i.target.id))
                new_else_stmts.append(assig_st)
                self.fwd_pass_tape.append(i.target.id)

                # increasing the counter....
                updated_st = loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                            val = loma_ir.BinaryOp(loma_ir.Add(),
                                                                    loma_ir.Var("tape_ctr"),
                                                                    loma_ir.ConstInt(1)))
                new_else_stmts.append(updated_st)
                new_else_stmts.append(i)

            return  loma_ir.IfElse(cond = new_cond,
                                   then_stmts = irmutator.flatten(new_then_stmts),
                                   else_stmts = irmutator.flatten(new_else_stmts))
                
        def mutate_while(self, node):
            
            new_cond = node.cond
            loop_var = node.cond.left
            new_while_body = []
            for i in node.body:

                if isinstance(i, loma_ir.While):
                    self.level_while+=1
                    new_while_body.append(loma_ir.Assign(target = loma_ir.Var('loop_tmp' + str(self.level_while)),
                                                         val = loma_ir.ConstInt(0)))
                    new_while_body.append(self.mutate_while(i))
                    new_while_body.append(loma_ir.Assign(target = loma_ir.ArrayAccess(loma_ir.Var('loop_ctr' + str(self.level_while)), 
                                                                        loma_ir.Var('loop_ptr' + str(self.level_while))),
                                                                        val = loma_ir.Var('loop_tmp' + str(self.level_while))))


                    new_while_body.append(loma_ir.Assign(target = loma_ir.Var('loop_ptr' + str(self.level_while)),
                                                    val = loma_ir.BinaryOp(loma_ir.Add(),
                                                                            loma_ir.Var('loop_ptr' + str(self.level_while)),
                                                                            loma_ir.ConstInt(1))))
                    self.level_while-=1
                    continue
                
                self.data_types_list.extend([i.target.t])
                self.input_var_cnt+=1

                if isinstance(i, loma_ir.Assign):
                    self.elements_inside_loop.append(i.target.id)

                if i.target.id not in self.loop_vars:
                    assig_st = loma_ir.Assign(target = loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                        loma_ir.Var("tape_ctr")),
                                                                        val = loma_ir.Var(i.target.id))
                    new_while_body.append(assig_st)
                    self.fwd_pass_tape.append(i.target.id)

                    # increasing the counter....
                    updated_st = loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                                val = loma_ir.BinaryOp(loma_ir.Add(),
                                                                        loma_ir.Var("tape_ctr"),
                                                                        loma_ir.ConstInt(1)))
                    new_while_body.append(updated_st)
                new_while_body.append(i)
            
            if self.level_while == 0:
                new_while_body.append(loma_ir.Assign(target = loma_ir.Var('loop_ctr' + str(0)),
                                                val = loma_ir.BinaryOp(loma_ir.Add(),
                                                                        loma_ir.Var('loop_ctr' + str(0)),
                                                                        loma_ir.ConstInt(1))))      
            else:
                new_while_body.append(loma_ir.Assign(target = loma_ir.Var('loop_tmp' + str(self.level_while)),
                                                val = loma_ir.BinaryOp(loma_ir.Add(),
                                                                        loma_ir.Var('loop_tmp' + str(self.level_while)),
                                                                        loma_ir.ConstInt(1))))


            self.loop_ctr_cnt += 1
            return loma_ir.While(cond = new_cond,
                                 body = new_while_body,
                                 max_iter = node.max_iter)
    

    def compute_loop_vars(node, loop_vars):
        for n in node.body:
            if isinstance(n, loma_ir.While):
                loop_vars.append(n.cond.left.id)
                compute_loop_vars(n, loop_vars)


    # Apply the differentiation.

    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW2: TODO

            # call Norm mut
            cnm = CallNormalizeMutator()
            node = cnm.mutate_function_def(node)
            self.call_mapping = cnm.call_var_to_func_mapping
            self.seen_vars = cnm.seen_variables
            self.parallel_status = node.is_simd
            
            # finding while loop variables..
            loop_vars = []
            compute_loop_vars(node, loop_vars)

            # looping the existing arguments...
            updated_arguments = []
            output_args = []
            for arg in node.args:
                
                # for input arguments...
                if arg.i == loma_ir.In():
                    # orginal argument passed as it is 
                    updated_arguments.append(arg)

                    # adjoint of the input argument
                    new_name = "_d" + arg.id
                    new_arg = loma_ir.Arg(new_name, arg.t, loma_ir.Out())
                    updated_arguments.append(new_arg)

                elif arg.i == loma_ir.Out():
                    updated_arguments.append(loma_ir.Arg(id = "_d" + arg.id,
                                                         t = arg.t,
                                                         i = loma_ir.In()))
                    output_args.append(arg.id)
            
            if node.ret_type != None:
                # updated return statement added
                new_arg = loma_ir.Arg("_dreturn", node.ret_type, loma_ir.In())
                updated_arguments.append(new_arg)

            # updating function body definition
            # fwd_pass
            elements_inside_loop = []
            number_of_lines = len(node.body)
            for n in node.body:
                if isinstance(n, loma_ir.While):
                    number_of_lines+= n.max_iter*3
                    break 


            fwd_pass = []
            fm = FwdPassMutator(count = number_of_lines, out_args = output_args, loop_vars = loop_vars, seen_vars = self.seen_vars)
            fm.mutate_function_def(node)
            fwd_pass = fm.dec_stmts
            dl_list = fm.data_types_list[::-1]
   
            # additional placeholder variables...
            tmp_decs  = []
            self.dt_cnt = []
            
            #tmp_decs.append(loma_ir.Declare("_adj_" + str(0), t = loma_ir.Float()))
            for i in range(fm.input_var_cnt):
                if isinstance(dl_list[i], loma_ir.Int):
                    dl_list[i] = loma_ir.Float()
                tmp_decs.append(loma_ir.Declare("_adj_" + str(i), t = dl_list[i]))
                self.dt_cnt.append(dl_list[i])
            
            
            # back_pass
            self.global_var_cnt = 0
            self.assign_flg = False
            self.update_var_holders = []
            self.tape_elements = fm.fwd_pass_tape
            self.back_pass = []
            self.flag_func = False
            self.nested_while_level = 0
            elements_inside_loop = fm.elements_inside_loop
            self.loop_vars = loop_vars
            self.parent_func = node.id
            self.adjoint = None
            
    
            for s in reversed(node.body):

                self.back_pass.append(self.mutate_stmt(s))
                                
                # popping out tape ops....
                if len(self.tape_elements) > 0:

                    var_name = self.tape_elements.pop()
                    if var_name not in elements_inside_loop:
            
                        self.back_pass.append(loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                                    val = loma_ir.BinaryOp(loma_ir.Sub(),
                                                                        loma_ir.Var("tape_ctr"),
                                                                        loma_ir.ConstInt(1))))

                        if isinstance(var_name, loma_ir.StructAccess):
                            self.back_pass.append(loma_ir.Assign(target =  var_name,            
                                                        val =  loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                                loma_ir.Var("tape_ctr"))))
                        else:
                            self.back_pass.append(loma_ir.Assign(target =  loma_ir.Var(var_name),            
                                                        val =  loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                                loma_ir.Var("tape_ctr"))))

            tmp_decs = []
            for i in range(fm.input_var_cnt):
                tmp_decs.append(loma_ir.Declare("_adj_" + str(i), t = self.dt_cnt[i]))
                self.dt_cnt.append(dl_list[i])

            # new function  
            updated_func = loma_ir.FunctionDef(diff_func_id,
                                               updated_arguments,
                                               irmutator.flatten(fwd_pass + tmp_decs + self.back_pass),
                                               node.is_simd,
                                               None)

            return updated_func

        def mutate_return(self, node):
            # HW2: TODO

            # global variable...
            self.adjoint = loma_ir.Var('_dreturn')
            stmts = self.mutate_expr(node.val)
            self.adjoint = None

            return stmts 

        def mutate_declare(self, node):
            # HW2: TODO
            stmts = []
            if node.val != None:
                self.adjoint = loma_ir.Var(f'_d{node.target}')
                stmts = self.mutate_expr(node.val)
                self.adjoint = None
            return stmts

        def mutate_assign(self, node):
            # HW2: TODO
            # if isinstance(node.target.t, loma_ir.Struct):
            #     return []

            if isinstance(node.target, loma_ir.ArrayAccess):
                self.adjoint = loma_ir.ArrayAccess(loma_ir.Var("_d" + node.target.array.id),
                                                   node.target.index)

            elif isinstance(node.target, loma_ir.StructAccess):
                self.adjoint = loma_ir.StructAccess(struct = loma_ir.Var(id = "_d" + node.target.struct.id,
                                                                        t = loma_ir.Struct(id = node.target.struct.t.id,
                                                                                           members = node.target.struct.t.members)),
                                                    member_id = node.target.member_id)

            else:
                self.adjoint = loma_ir.Var(f'_d{node.target.id}', t = node.target.t)
            
            self.assign_flg = True
            self.func_updates = False
            right_mutations = self.mutate_expr(node.val)
            self.assign_flg = False

            # zero out gradient values....
            if isinstance(node.target, loma_ir.ArrayAccess):
                flg = True
                for n_a in funcs[self.parent_func].args:
                    if n_a.i == loma_ir.Out():
                        if n_a.id == node.target.array.id:
                            flg = False
                            break

                if flg:
                    right_mutations.append(loma_ir.Assign(target = loma_ir.ArrayAccess(loma_ir.Var("_d" + node.target.array.id),
                                                                                    node.target.index),
                                                        val = loma_ir.ConstFloat(0.0)))
            elif isinstance(node.target, loma_ir.StructAccess):
                right_mutations.append(loma_ir.Assign(target = loma_ir.StructAccess(struct = loma_ir.Var(id = "_d" + node.target.struct.id,
                                                                                                        t = loma_ir.Struct(id = node.target.struct.t.id,
                                                                                                                            members = node.target.struct.t.members)),
                                                                                    member_id = node.target.member_id),
                                                    val = loma_ir.ConstFloat(0.0)))
            elif isinstance(node.target.t, loma_ir.Struct):
                update_n = loma_ir.Assign(target = loma_ir.Var("_d" + node.target.id , t = node.target.t),
                                          val = node.val)
                right_mutations.append(assign_zero(update_n.target))
            else:
                if not self.func_updates:
                    # flg = True
                    # if funcs.get(self.parent_func, "") != "":
                    #     for n_a in funcs[self.parent_func].args:
                    #         if n_a.i == loma_ir.Out():
                    #             if n_a.id == node.target.id:
                    #                 flg = False
                    #                 break
                    # if flg:
                    right_mutations.append(loma_ir.Assign(target = loma_ir.Var(f'_d{node.target.id}'),
                                                        val = loma_ir.ConstFloat(0.0)))
        
            if not self.func_updates:
                # delayed var updates....
                for ele in self.update_var_holders:
                    if isinstance(ele[1].t, loma_ir.Struct):
                        right_mutations.append(loma_ir.Assign(ele[1],
                                                            loma_ir.Var(ele[0])))
                    else:
                        if self.parallel_status:
                            right_mutations.append(loma_ir.CallStmt(loma_ir.Call('atomic_add', [ele[1], loma_ir.Var(ele[0])])))
                        else:
                            right_mutations.append(loma_ir.Assign(ele[1],
                                                                loma_ir.BinaryOp(loma_ir.Add(),
                                                                                ele[1],
                                                                                loma_ir.Var(ele[0]))))
            
     
            # clearing the updates...
            self.update_var_holders = []
            self.adjoint = None
            self.func_updates =  False
            return right_mutations
                        
        def mutate_ifelse(self, node):
            # HW3: 
            l_i = []
            for c_ival,i in enumerate(node.then_stmts[::-1]):

                if isinstance(i, loma_ir.IfElse):
                    l_i.append(self.mutate_stmt(i))
                else:

                    if c_ival != 0:
                        var_name = self.tape_elements.pop()
                    
                        l_i.append(loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                                    val = loma_ir.BinaryOp(loma_ir.Sub(),
                                                                        loma_ir.Var("tape_ctr"),
                                                                        loma_ir.ConstInt(1))))

                        if isinstance(var_name, loma_ir.StructAccess):
                            l_i.append(loma_ir.Assign(target =  var_name,            
                                                        val =  loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                                loma_ir.Var("tape_ctr"))))
                        else:
                            l_i.append(loma_ir.Assign(target =  loma_ir.Var(var_name),            
                                                        val =  loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                                loma_ir.Var("tape_ctr"))))
                    
                    l_i.append(self.mutate_stmt(i))
                
            r_i = []
            for c_ival,i in enumerate(node.else_stmts[::-1]):

                if isinstance(i, loma_ir.IfElse):
                    l_i.append(self.mutate_stmt(i))
                else:

                    if c_ival != 0:
                        var_name = self.tape_elements.pop()
                    
                        r_i.append(loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                                    val = loma_ir.BinaryOp(loma_ir.Sub(),
                                                                        loma_ir.Var("tape_ctr"),
                                                                        loma_ir.ConstInt(1))))

                        if isinstance(var_name, loma_ir.StructAccess):
                            r_i.append(loma_ir.Assign(target =  var_name,            
                                                        val =  loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                                loma_ir.Var("tape_ctr"))))
                        else:
                            r_i.append(loma_ir.Assign(target =  loma_ir.Var(var_name),            
                                                        val =  loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                                loma_ir.Var("tape_ctr"))))
                    
                    r_i.append(self.mutate_stmt(i))
       
            updated_if_else = loma_ir.IfElse(cond = node.cond,
                                             then_stmts = irmutator.flatten(l_i),
                                             else_stmts = irmutator.flatten(r_i))

            return updated_if_else

        def mutate_call_stmt(self, node):
            # HW3
            tmp = self.parent_func
            self.parent_func = node.call.id
            self.func_updates = True
            op = self.mutate_expr(node.call)
            self.func_updates = False
            self.parent_func = tmp
            return op
        
        def mutate_while(self, node):
            # HW3: TODO

            updated_lines = []
            if self.nested_while_level == 0:
                loop_var = loma_ir.Var('loop_ctr' + str(0))
            else:
                loop_var = loma_ir.Var('loop_tmp' + str(self.nested_while_level))


            for c_ival,i in enumerate(node.body[::-1]):
                if c_ival != 0:
                    if len(self.tape_elements) > 0:
                        var_name = self.tape_elements.pop()
                    
                        updated_lines.append(loma_ir.Assign(target = loma_ir.Var("tape_ctr"),
                                                    val = loma_ir.BinaryOp(loma_ir.Sub(),
                                                                        loma_ir.Var("tape_ctr"),
                                                                        loma_ir.ConstInt(1))))

                        if isinstance(var_name, loma_ir.StructAccess):
                            updated_lines.append(loma_ir.Assign(target =  var_name,            
                                                        val =  loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                                loma_ir.Var("tape_ctr"))))
                        else:
                            updated_lines.append(loma_ir.Assign(target =  loma_ir.Var(var_name),            
                                                        val =  loma_ir.ArrayAccess(loma_ir.Var("tape"), 
                                                                                loma_ir.Var("tape_ctr"))))

                    if isinstance(i, loma_ir.While):
                        self.nested_while_level+=1
                        updated_lines.append(loma_ir.Assign(target = loma_ir.Var('loop_ptr' + str(self.nested_while_level)),
                                                    val = loma_ir.BinaryOp(loma_ir.Sub(),
                                                                        loma_ir.Var('loop_ptr' + str(self.nested_while_level)),
                                                                        loma_ir.ConstInt(1))))
                                                        
                        updated_lines.append(loma_ir.Assign(target =  loma_ir.Var('loop_tmp' + str(self.nested_while_level)),            
                                                            val =  loma_ir.ArrayAccess(loma_ir.Var('loop_ctr' + str(self.nested_while_level)), 
                                                                                    loma_ir.Var('loop_ptr' + str(self.nested_while_level)))))

                    if (isinstance(i, loma_ir.Assign)):
                        if i.target.id in self.loop_vars:
                            continue
                    updated_lines.append(self.mutate_stmt(i))

    
                    updated_lines.append(loma_ir.Assign(target = loop_var,
                                                val = loma_ir.BinaryOp(loma_ir.Sub(),
                                                                    loop_var,
                                                                    loma_ir.ConstInt(1))))


        
            updated_while = loma_ir.While(cond = loma_ir.BinaryOp(op = loma_ir.Greater(), 
                                                                  left  = loop_var, 
                                                                  right = loma_ir.ConstInt(0)),
                                          max_iter = node.max_iter,
                                          body = irmutator.flatten(updated_lines))
                
            self.nested_while_level-=1
            return updated_while

        def mutate_const_float(self, node):
            # HW2: TODO
            return []

        def mutate_const_int(self, node):
            # HW2: TODO
            return []

        def mutate_var(self, node):

            # HW2: TODO

            if self.assign_flg:
                #tmp_dec = loma_ir.Declare("_adj_" + str(self.global_var_cnt), t = self.adjoint.t)
                stmts = [loma_ir.Assign(loma_ir.Var('_adj_' + str(self.global_var_cnt)),self.adjoint)]
                up_name = node.id
                if "_d" not in node.id:
                    up_name = "_d" + up_name

                
                self.update_var_holders.append(('_adj_' + str(self.global_var_cnt), loma_ir.Var(up_name, t = node.t)))
                if isinstance(node.t, loma_ir.Struct):
                    self.dt_cnt[self.global_var_cnt] = node.t
                self.global_var_cnt+=1
            else:
                if isinstance(node.t, loma_ir.Struct):
                    stmts = [loma_ir.Assign(loma_ir.Var('_d' + node.id, t = node.t),
                                            self.adjoint)]
                    self.global_var_cnt+=1
                else:
                    stmts = [loma_ir.Assign(loma_ir.Var('_d' + node.id),
                                        loma_ir.BinaryOp(loma_ir.Add(),
                                                            loma_ir.Var('_d' + node.id, t = node.t),
                                                            self.adjoint))]

           
            return stmts

        def mutate_array_access(self, node):
            # HW2: TODO
            # naming d variable...
            tmp_holder_name = loma_ir.ArrayAccess
            if not isinstance(node.array, loma_ir.Var):
                tmp_holder_name =  loma_ir.ArrayAccess(loma_ir.ArrayAccess(loma_ir.Var("_d" + node.array.array.id, 
                                                                                      t =  node.array.array.t),
                                                                           node.array.index, 
                                                                           t = node.array.t),
                                                       node.index,
                                                       t = node.t)
                
            else:
                tmp_holder_name = loma_ir.ArrayAccess(loma_ir.Var("_d" + node.array.id),
                                                      node.index)
                

            
            if self.assign_flg:
 
                #tmp_dec = loma_ir.Declare("_adj_" + str(self.global_var_cnt), t = self.adjoint.t)
                stmts = [loma_ir.Assign(loma_ir.Var('_adj_' + str(self.global_var_cnt)), self.adjoint)]
                self.update_var_holders.append(('_adj_' + str(self.global_var_cnt), 
                                                tmp_holder_name))
                self.global_var_cnt+=1
            else:
                stmts = [loma_ir.Assign(tmp_holder_name,
                                        loma_ir.BinaryOp(loma_ir.Add(),
                                                        tmp_holder_name,
                                                        self.adjoint))]

            return stmts

        def mutate_struct_access(self, node):
            # HW2: TODO
            tmp_holder_name = loma_ir.StructAccess(struct = loma_ir.Var(id = "_d" + node.struct.id,
                                                                        t = loma_ir.Struct(id = node.struct.t.id,
                                                                                           members = node.struct.t.members)),
                                                   member_id = node.member_id)

            if self.assign_flg:
                #tmp_dec = loma_ir.Declare("_adj_" + str(self.global_var_cnt), t = self.adjoint.t)
                stmts = [loma_ir.Assign(loma_ir.Var('_adj_' + str(self.global_var_cnt)), self.adjoint)]
                self.update_var_holders.append(('_adj_' + str(self.global_var_cnt), 
                                                tmp_holder_name))
                self.global_var_cnt+=1
            else:

                stmts = [loma_ir.Assign(tmp_holder_name,
                                        loma_ir.BinaryOp(loma_ir.Add(),
                                                        tmp_holder_name,
                                                        self.adjoint))]

            return stmts

        def mutate_add(self, node):
            # HW2: TODO
            l = self.mutate_expr(node.left)
            r = self.mutate_expr(node.right)

            
            return l+r

        def mutate_sub(self, node):
            # HW2: TODO
            l = self.mutate_expr(node.left)
            og_adjoint = self.adjoint

            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                            loma_ir.ConstInt(-1),
                                            self.adjoint)
            r = self.mutate_expr(node.right)
            self.adjoint = og_adjoint

            return l + r

        def mutate_mul(self, node):
            # HW2: TODO
            og_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                            self.adjoint,
                                            node.right)
            l = self.mutate_expr(node.left)
            self.adjoint = og_adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                            self.adjoint,
                                            node.left)
            r = self.mutate_expr(node.right)
            self.adjoint = og_adjoint

            return l+r
            
        def mutate_div(self, node):
            # HW2: TODO
            og_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Div(),
                                            self.adjoint,
                                            node.right)
            l = self.mutate_expr(node.left)
            self.adjoint = og_adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                            self.adjoint,
                                            loma_ir.ConstInt(-1))
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                            self.adjoint,
                                            node.left)
            sq_dem = loma_ir.BinaryOp(loma_ir.Mul(),
                                       node.right,
                                       node.right)
            self.adjoint = loma_ir.BinaryOp(loma_ir.Div(),
                                            self.adjoint,
                                            sq_dem)
            
            r = self.mutate_expr(node.right)

            self.adjoint = og_adjoint

            return l+r

        def mutate_call(self, node):
            # HW2: TODO
            stmts = []
            match node:
                case loma_ir.Call('sin'):
                    og_adjoint = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                                     self.adjoint, 
                                                     loma_ir.Call('cos', node.args))
                    for i in node.args:
                        stmts.append(self.mutate_expr(i))

                    self.adjoint = og_adjoint
                
                case loma_ir.Call('cos'):
                    og_adjoint = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                                     self.adjoint, 
                                                     loma_ir.BinaryOp(loma_ir.Mul(),
                                                                      loma_ir.ConstInt(-1),
                                                                      loma_ir.Call('sin', node.args)))
                    for i in node.args:
                        stmts.append(self.mutate_expr(i))

                    self.adjoint = og_adjoint

                case loma_ir.Call('sqrt'):
                    og_adjoint = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Div(),
                                                     self.adjoint, 
                                                     loma_ir.BinaryOp(loma_ir.Mul(),
                                                                      loma_ir.ConstInt(2),
                                                                      loma_ir.Call('sqrt', node.args)))
                    for i in node.args:
                        stmts.append(self.mutate_expr(i))

                    self.adjoint = og_adjoint
                
                case loma_ir.Call('pow'):
                    og_adjoint = self.adjoint
                    # base....
                    tmp_2 = loma_ir.Call('pow', [node.args[0], loma_ir.BinaryOp(loma_ir.Sub(),
                                                                                node.args[1],
                                                                                loma_ir.ConstFloat(1.0))])
                    tmp_1 = loma_ir.BinaryOp(loma_ir.Mul(),
                                             node.args[1],
                                             tmp_2)

                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                                     self.adjoint, 
                                                     tmp_1)
                    stmts.append(self.mutate_expr(node.args[0]))

                    # power
                    self.adjoint = og_adjoint
                    tmp_3 = loma_ir.BinaryOp(loma_ir.Mul(),
                                                     loma_ir.Call('log', [node.args[0]]), 
                                                     node)
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                                     self.adjoint, 
                                                     tmp_3)
                    stmts.append(self.mutate_expr(node.args[1]))
                    self.adjoint = og_adjoint
                
                case loma_ir.Call('exp'):
                    og_adjoint = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(),
                                                     self.adjoint, 
                                                     node)
                    for i in node.args:
                        stmts.append(self.mutate_expr(i))

                    self.adjoint = og_adjoint
                
                case loma_ir.Call('log'):
                    og_adjoint = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Div(),
                                                     self.adjoint, 
                                                     node.args[0])
                    for i in node.args:
                        stmts.append(self.mutate_expr(i))

                    self.adjoint = og_adjoint

                case loma_ir.Call('float2int'):
                    og_adjoint = self.adjoint
                    self.adjoint = loma_ir.ConstInt(0)

                    for i in node.args:
                        stmts.append(self.mutate_expr(i))

                    self.adjoint = og_adjoint
                
                case loma_ir.Call('int2float'):
                    og_adjoint = self.adjoint
                    self.adjoint = loma_ir.ConstFloat(0.0)

                    for i in node.args:
                        stmts.append(self.mutate_expr(i))

                    self.adjoint = og_adjoint
                
                case loma_ir.Call('atomic_add'):
                    expression = loma_ir.Assign(node.args[0],
                                               loma_ir.BinaryOp(loma_ir.Add(), 
                                                                node.args[0], 
                                                                node.args[1]))
                    
                    og_adjoint = self.adjoint
                    self.adjoint = loma_ir.Var("_d" + node.args[0].id)
                    left = loma_ir.ArrayAccess(loma_ir.Var("_d" + node.args[1].array.id),
                                               node.args[1].index)

                    right = loma_ir.Var("_d" +  node.args[0].id)

                    
                    stmts.append(loma_ir.CallStmt(loma_ir.Call('atomic_add', 
                                                               [left, right])))
                    self.adjoint = og_adjoint


                case _:

                    # reverse mode mapping....
                    if func_to_rev.get(node.id,"") == "":
                        return []

                    self.func_updates = True

                    if self.assign_flg:
                        stmts.append([loma_ir.Assign(loma_ir.Var('_adj_' + str(self.global_var_cnt)),self.adjoint)])
                        stmts.append([loma_ir.Assign(loma_ir.Var(self.adjoint.id),loma_ir.ConstFloat(0.0))])
                        self.update_var_holders.append(('_adj_' + str(self.global_var_cnt), self.adjoint))
                        self.adjoint = loma_ir.Var('_adj_' + str(self.global_var_cnt))
                        self.global_var_cnt+=1
                    

                    
                    new_node_id = func_to_rev[node.id]

                    # argument list....
                    argument_props = funcs[node.id].args

                    # original primal function definition...
                    og_func_def = funcs[node.id]

                    # updating the arguments....
                    updated_arguments = []
                    call_args = []
                    for c, arg in enumerate(argument_props):
                        # for input arguments...
                        if arg.i == loma_ir.In():
                            # orginal argument passed as it is 
                            updated_arguments.append(arg)
                            
                            call_args.append(node.args[c])

                            # adjoint of the input argument
                            new_name = "_d" + node.args[c].id
                            new_arg = loma_ir.Arg(new_name, arg.t, loma_ir.Out())
                            updated_arguments.append(new_arg)
                            call_args.append(loma_ir.Var(new_name))
                        else:
                            # updated_arguments.append(loma_ir.Arg(id = "_d" + arg.id,
                            #                              t = arg.t,
                            #                              i = loma_ir.In()))
                            # call_args.append(loma_ir.Var(id = "_d" + arg.id,
                            #                              t = node.args[c].t))
                        
                            varName_arg = "_d" + node.args[c].id
                            # if arg.i == loma_ir.Out():
                            #     stmts.append(loma_ir.Declare("_d" + varName_arg, t = loma_ir.Float()))
                            #     stmts.append(loma_ir.Assign(loma_ir.Var("_d" + varName_arg),
                            #                                 loma_ir.Var(varName_arg)))
                            #     varName_arg = "_d" + varName_arg

                           
               
                            updated_arguments.append(loma_ir.Arg(id = varName_arg,
                                                        t = arg.t,
                                                        i = loma_ir.In()))
                            call_args.append(loma_ir.Var(id = varName_arg,
                                                        t = node.args[c].t))
                          

                           
                    if og_func_def.ret_type != None:
                        # updated return statement added
                        new_arg = loma_ir.Arg("_dreturn", og_func_def.ret_type, loma_ir.In())
                        updated_arguments.append(new_arg)
 
                        call_args.append(self.adjoint)

                    # updating the body....
                    updated_body = []
                    for i in og_func_def.body:
                        updated_body.append(self.mutate_stmt(i))
                    
                    # Mutated Function.....
                    new_rev_diff = loma_ir.FunctionDef(new_node_id, 
                                                       updated_arguments,
                                                       irmutator.flatten(updated_body),
                                                       og_func_def.is_simd,
                                                       None)

                    stmts.append(loma_ir.CallStmt(loma_ir.Call(new_node_id, call_args)))
                    
                    for c,n in enumerate(node.args):
                        if (argument_props[c].i == loma_ir.Out()) & (isinstance(n.t, loma_ir.Float)):
                            stmts.append(loma_ir.Assign(loma_ir.Var("_d" + n.id),
                                                        loma_ir.ConstFloat(0.0)))
                        # if n.id == "y0":
                           
                        #     stmts.append(loma_ir.Assign(loma_ir.Var("_dy0"),
                        #                                 loma_ir.ConstFloat(0.0)))




            return stmts

 
    return RevDiffMutator().mutate_function_def(func)

 
