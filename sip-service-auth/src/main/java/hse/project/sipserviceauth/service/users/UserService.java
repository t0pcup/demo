package hse.project.sipserviceauth.service.users;

import hse.project.sipserviceauth.exception.ApiRequestException;
import hse.project.sipserviceauth.model.domain.User;
import hse.project.sipserviceauth.model.request.UserRequest;
import hse.project.sipserviceauth.model.response.CreateResponse;
import hse.project.sipserviceauth.model.response.DeleteResponse;
import hse.project.sipserviceauth.model.response.UpdateResponse;
import hse.project.sipserviceauth.repository.TokenRepository;
import hse.project.sipserviceauth.repository.UserRepository;
import hse.project.sipserviceauth.service.CrudService;

import lombok.RequiredArgsConstructor;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class UserService implements CrudService<User, UserRequest> {

    private final UserRepository userRepository;

    private final PasswordEncoder passwordEncoder;
    private final TokenRepository tokenRepository;

    @Override
    public CreateResponse<User> create(UserRequest userRequest) throws ApiRequestException {
        User user = User.builder()
                .username(userRequest.getUsername())
                .password(passwordEncoder.encode(userRequest.getPassword()))
                .name(userRequest.getName())
                .surname(userRequest.getSurname())
                .patronymic(userRequest.getPatronymic())
                .role(userRequest.getRole())
                .build();

        userRepository.save(user);

        return new CreateResponse<>("New user created!", user);
    }

    @Override
    public List<User> readAll() {
        return userRepository.findAll();
    }

    @Override
    public User readById(UUID userId) {
        return userRepository.findById(userId).orElse(null);
    }

    @Override
    public UpdateResponse<User> updateById(UUID id, UserRequest updated) throws ApiRequestException {
        //TODO: implement update user by id method
        return null;
    }

    @Override
    public DeleteResponse<User> deleteById(UUID userId) {
        if (userRepository.existsById(userId)) {
            userRepository.deleteById(userId);
            return new DeleteResponse<>("User deleted!", userId);
        }

        return null;
    }
}
