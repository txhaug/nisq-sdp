
%function a = matlabSDP(b)
%    %disp("test")
%    a=5;
%end

function beta_result=matlabSDP(E_matrix_in,D_matrix_in,shift_value,do_conserved,conserved_number,C_matrix_in,C2_matrix_in)
    E_matrix=E_matrix_in;
    D_matrix=D_matrix_in;
    C_matrix=C_matrix_in;
    C2_matrix=C2_matrix_in;
    n_expand_states=size(E_matrix);
    n_expand_states=n_expand_states(1);
    
    D_matrix_sdp=D_matrix+shift_value*E_matrix;
    if(do_conserved==false)
        cvx_begin
            variable beta_q(n_expand_states,n_expand_states) hermitian semidefinite;
            obj = real(trace(beta_q*D_matrix_sdp));
            minimize obj
            subject to
                real(trace(beta_q*E_matrix)) <= 1;
        cvx_end
    else
        cvx_begin
            variable beta_q(n_expand_states,n_expand_states) hermitian semidefinite;
            obj = real(trace(beta_q*D_matrix_sdp));
            minimize obj
            subject to
                real(trace(beta_q*E_matrix)) <= 1;
                real(trace(beta_q*C_matrix)) == conserved_number;
                real(trace(beta_q*C2_matrix)) <= (conserved_number^2);
                real(trace(beta_q*C2_matrix)) >= (conserved_number^2);
        cvx_end

    end
    
    beta_result=beta_q;
end

%norm=trace(beta_q*E_matrix)
%energy=trace(beta_q*D_matrix)

%n_state_list(k)=n_expand_states;
%energy_list(k)=real(energy);
