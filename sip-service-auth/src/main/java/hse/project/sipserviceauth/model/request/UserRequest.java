package hse.project.sipserviceauth.model.request;

import hse.project.sipserviceauth.utils.Role;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UserRequest {

    private String username;

    private String password;

    private String name;

    private String surname;

    private String patronymic;

    private Role role;
}
