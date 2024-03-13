# Node specification
struct Diode end

@node Diode Deterministic [out,inp]
 
# Rule specification
@rule Diode(:out, Marginalisation) (inp::Any) = begin
    return inp
end 

@rule Diode(:inp, Marginalisation) (out::Any) = begin
    return Uninformative()
end 

