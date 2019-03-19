function pd_torques(nlp, bounds)
    %VIRTUAL_CONSTRAINTS Summary of this function goes here
    %   Detailed explanation goes here
    
    % relative degree 2 outputs
    
    t = SymVariable('t');
    k = SymVariable('k');
    T  = SymVariable('t',[2,1]);
    nNode = SymVariable('nNode');
    tsubs = T(1) + ((k-1)./(nNode-1)).*(T(2)-T(1));
    ya = output.ActualFuncs{1};
    yd = output.DesiredFuncs{1};
    dya = output.ActualFuncs{2};
    dyd = output.DesiredFuncs{2};
    u = domain.Inputs.Control.u;
    kp = bounds.gains.kp';
    kd = bounds.gains.kd';
    expr = u + transpose(kp.*transpose(ya - yd)) + transpose(kd.*transpose(dya-dyd));
    expr_s = subs(expr,t,tsubs);
    x = domain.States.x;
    dx = domain.States.dx;
    a = {SymVariable(tomatrix(output.OutputParams(:)))};
    a_name = output.OutputParamName;
    p = {SymVariable(tomatrix(output.PhaseParams(:)))};
    p_name = output.PhaseParamName;
    fun = SymFunction(['pd_feedback_',domain.Name],expr_s,{T,x,dx,u,a{1},p{1}},{k,nNode});
    for i=1:nlp.NumNode
        addNodeConstraint(nlp, fun, [{'T','x','dx','u'},a_name, p_name], i, 0, 0, 'Nonlinear',{i,nlp.NumNode});
    end
    
end

