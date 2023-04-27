package hse.project.sipserviceauth.utils;

import hse.project.sipserviceauth.model.domain.User;

import org.springframework.security.core.context.SecurityContextHolder;

public class AuthorizedUser {

    public static User getUser() {
        Object principal = SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        return principal instanceof User ? (User) principal : null;
    }
}
