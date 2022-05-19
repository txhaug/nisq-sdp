function beta_result=matlabPOVM_EMatrix_SDP(E_matrix_in,R_matrix_in,slack_overlap)
    %R_matrix is the positive semidifiante matrix describing the states to
    %be descriminated
    % beta is the POVM matrix
    E_matrix=E_matrix_in;
    R_matrix=R_matrix_in;
    n_states=size(R_matrix); %number of states to be discrimanted
    n_states=n_states(3);
    n_expand_states=size(E_matrix);
    n_expand_states=n_expand_states(1);
    
    cvx_begin
        variable var_beta(n_expand_states,n_expand_states,n_states) hermitian semidefinite;
        obj = 0;
        for i=1:n_states
            obj= obj + real(trace(var_beta(:,:,i)*E_matrix*R_matrix(:,:,i)*E_matrix));
        end
        maximize obj
        subject to
            %for i=1:n_states
            %    for j=1:n_states
            %        if(i~=j)
            %            real(trace(var_beta(:,:,i)*E_matrix*R_matrix(:,:,j)*E_matrix)) <= slack_overlap;
            %        end
            %    end
            %end
            expression unambig(n_states);
            for i=1:n_states % sum over POVM
                unambig(i)=0;
                for j=1:n_states % sum over states
                    if(i~=j)
                        unambig(i)=unambig(i)+real(trace(var_beta(:,:,i)*E_matrix*R_matrix(:,:,j)*E_matrix));
                    end
                end
                unambig(i)<= slack_overlap;
            end

            sum_beta=var_beta(:,:,1)
            for i=2:n_states
                sum_beta=sum_beta+var_beta(:,:,i)
            end
            E_matrix-E_matrix*sum_beta*E_matrix==hermitian_semidefinite(n_expand_states)

    cvx_end

    beta_result=var_beta;
end