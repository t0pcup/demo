package hse.project.sipserviceauth.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.UUID;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class DeleteResponse<T> {

    private String message;
    
    private UUID deletedObjectId;
}
