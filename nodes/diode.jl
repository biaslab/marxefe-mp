
struct Diode end
@node Diode Deterministic [out,in]
 

@rule Diode(:out, Marginalisation) (q_in::Any,) = begin
    return q_in
end 

@rule Diode(:in, Marginalisation) (q_out::Any,) = begin
    return Uninformative
end 

