package hse.project.sipserviceauth.api.v1;

import hse.project.sipserviceauth.model.domain.User;
import hse.project.sipserviceauth.model.request.RegisterRequest;
import hse.project.sipserviceauth.model.response.DeleteResponse;
import hse.project.sipserviceauth.service.users.UserService;
import hse.project.sipserviceauth.utils.AuthorizedUser;

import lombok.RequiredArgsConstructor;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping(path = "/me")
@CrossOrigin(origins = "http://127.0.0.1:5173")
@PreAuthorize("hasAnyRole('ROLE_USER', 'ROLE_ADMIN')")
@RequiredArgsConstructor
public class MyController {

    private final UserService userService;

    @GetMapping()
    public ResponseEntity<?> read() {
        User user = AuthorizedUser.getUser();

        if (user == null) {
            return new ResponseEntity<>("You are NOT authorized!", HttpStatus.UNAUTHORIZED);
        }

        return new ResponseEntity<>(user, HttpStatus.OK);
    }

    @PutMapping()
    public ResponseEntity<?> update(@RequestBody RegisterRequest request) {
        //TODO: Implement update my info
        return new ResponseEntity<>(request, HttpStatus.OK);
    }

    @DeleteMapping()
    public ResponseEntity<?> delete() {
        User user = AuthorizedUser.getUser();

        if (user == null) {
            return new ResponseEntity<>("You are NOT authorized!", HttpStatus.UNAUTHORIZED);
        }

        final DeleteResponse<User> deleteResponse = userService.deleteById(user.getUser_id());

        return deleteResponse != null
                ? new ResponseEntity<>(deleteResponse, HttpStatus.OK)
                : new ResponseEntity<>("User with such id not found!", HttpStatus.NOT_FOUND);
    }
}
