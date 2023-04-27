package hse.project.sipserviceauth.api.v1;

import hse.project.sipserviceauth.api.CrudController;
import hse.project.sipserviceauth.exception.ApiRequestException;
import hse.project.sipserviceauth.model.domain.User;
import hse.project.sipserviceauth.model.request.UserRequest;
import hse.project.sipserviceauth.model.response.CreateResponse;
import hse.project.sipserviceauth.model.response.DeleteResponse;
import hse.project.sipserviceauth.service.users.UserService;

import lombok.RequiredArgsConstructor;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@CrossOrigin(origins = "http://127.0.0.1:5173")
@PreAuthorize("hasAnyRole('ROLE_ADMIN')")
@RequiredArgsConstructor
public class UserController implements CrudController<UserRequest> {

    private final UserService userService;

    @Override
    @PostMapping("/user")
    public ResponseEntity<?> create(@RequestBody UserRequest userRequest) {
        CreateResponse<User> createResponse;

        try {
            createResponse = userService.create(userRequest);
        } catch (ApiRequestException e) {
            return new ResponseEntity<>(e.getMessage(), HttpStatus.BAD_REQUEST);
        }

        return new ResponseEntity<>(createResponse, HttpStatus.CREATED);
    }

    @Override
    @GetMapping("/users")
    public ResponseEntity<?> readAll() {
        final List<User> users = userService.readAll();

        return users != null
                ? new ResponseEntity<>(users, HttpStatus.OK)
                : new ResponseEntity<>("There are NO users!", HttpStatus.NOT_FOUND);
    }

    @Override
    @GetMapping("/user")
    public ResponseEntity<?> readById(@RequestParam("user_id") UUID userId) {
        final User user = userService.readById(userId);

        return user != null
                ? new ResponseEntity<>(user, HttpStatus.OK)
                : new ResponseEntity<>("No user with such id!", HttpStatus.NOT_FOUND);
    }

    @Override
    @PutMapping("/user")
    public ResponseEntity<?> updateById(@RequestParam("user_id") UUID userId, UserRequest updateUserRequest) {
        //TODO: UserController - Change User update method
        try {
            return userService.updateById(userId, updateUserRequest) != null
                    ? new ResponseEntity<>("Пользователь успешно обновлен", HttpStatus.OK)
                    : new ResponseEntity<>("Не удалось найти пользователя", HttpStatus.NOT_FOUND);
        } catch (ApiRequestException e) {
            return new ResponseEntity<>(e.getMessage(), HttpStatus.BAD_REQUEST);
        }
    }

    @Override
    @DeleteMapping("/user")
    public ResponseEntity<?> deleteById(@RequestParam("user_id") UUID userId) {
        final DeleteResponse<User> deleteResponse = userService.deleteById(userId);

        return deleteResponse != null
                ? new ResponseEntity<>(deleteResponse, HttpStatus.OK)
                : new ResponseEntity<>("No user with such id!", HttpStatus.NOT_FOUND);
    }
}
